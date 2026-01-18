from milp.sat_with_sequenced_vehicles import solve_cvrptw_milp_sat_with_sequenced_vehicles
from ortools.sat.cp_model_pb2 import CpSolverStatus
from typing import Union
from milp.cp_sat_2d import solve_cvrptw_milp_cp_sat_2d
from ortools.sat.python import cp_model
from milp.gemini_scip import solve_cvrptw_milp_gemini_scip
from milp.scip import solve_cvrptw_milp_scip
from milp.scip_with_load_vars import solve_cvrptw_milp_scip_with_load_vars
from milp.sat_with_load_vars import solve_cvrptw_milp_sat_with_load_vars
from milp.sat_2d import solve_cvrptw_milp_sat_2d
from milp.sat import solve_cvrptw_milp_sat
from typing import Any
from typing import Callable
from pathlib import Path
from dataclasses import dataclass
import time
import math
from typing import List, Tuple, Dict, Optional
from ortools.linear_solver import pywraplp

from parse import read_solomon_vrptw, VehicleInfo, Customer
from milp.utils import *


SolverCallable = Callable[
    [List[Customer], VehicleInfo, int, int],
    Tuple[
        Union[pywraplp.Solver, cp_model.CpSolver, None], # Chấp nhận cả 2 loại Solver
        Union[int, CpSolverStatus],                      # Chấp nhận cả int (MILP) và CpSolverStatus (CP-SAT)
        Dict[Any, Any],                                  # Chấp nhận dict biến của cả 2
        List[Customer]
    ]
]
# --- 1. CONFIGURATION ---
@dataclass
class BenchmarkConfig:
    input_folder: Path = Path("testcases")
    output_dir: Path = Path("results_milp")
    output_name: str = "benchmark_results.txt"
    limit_nodes: int = 15
    time_limit_sec: int = 30
    verbose: bool = False

@dataclass
class BenchmarkResult:
    instance_name: str
    status: str
    duration: float
    obj_value: str
    routes: List[List[int]]
    is_valid: bool
    validation_msg: str


def validate_solution(
    routes: List[List[int]],
    customers: List[Customer],
    vehicle: VehicleInfo,
    verbose: bool = False
) -> Tuple[bool, str]:
    """Kiểm tra tính hợp lệ và in chi tiết nếu verbose=True."""
    visited_customers = set()
    total_cost = 0.0

    if verbose:
        print("\n" + "="*80)
        print(f"{'VALIDATION DETAIL REPORT':^80}")
        print("="*80)

    for r_idx, route in enumerate(routes):
        if not route: continue

        # 1. Check Structure
        if route[0] != 0 or route[-1] != 0:
            return False, f"Route {r_idx} Error: Must start/end at Depot (0). Got: {route}"

        current_load = 0
        current_time = customers[0].ready_time
        route_dist = 0.0

        if verbose:
            print(f"\nVehicle {r_idx + 1} (Cap: {vehicle.capacity}): {route}")
            print(f"  {'Node':<5} | {'Load':<8} | {'Travel':<8} | {'Arrive':<8} | {'Wait':<6} | {'Start':<8} | {'Window':<12}")
            print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*12}")
            print(f"  {'0(D)':<5} | {'0':<8} | {'0.0':<8} | {'-':<8} | {'-':<6} | {current_time:<8.1f} | [{customers[0].ready_time}-{customers[0].due_date}]")

        for i in range(1, len(route)):
            prev = customers[route[i-1]]
            curr = customers[route[i]]

            # 2. Check Capacity
            current_load += curr.demand
            if current_load > vehicle.capacity:
                msg = f"Overload at Node {route[i]}. Load {current_load} > {vehicle.capacity}"
                if verbose: print(f"  >>> FAILED: {msg}")
                return False, msg

            # 3. Check Time & Window
            dist = manhattan(prev, curr)
            route_dist += dist
            arrival_time = current_time + prev.service_time + dist

            # Check Due Date
            if arrival_time > curr.due_date:
                msg = f"Late at Node {route[i]}. Arr {arrival_time:.1f} > Due {curr.due_date}"
                if verbose: print(f"  >>> FAILED: {msg}")
                return False, msg

            wait_time = max(0, curr.ready_time - arrival_time)
            start_service = max(arrival_time, curr.ready_time)

            if verbose:
                win_lbl = f"[{curr.ready_time}-{curr.due_date}]"
                print(f"  {route[i]:<5} | {current_load:<3}/{vehicle.capacity:<4} | {dist:<8.1f} | {arrival_time:<8.1f} | {wait_time:<6.1f} | {start_service:<8.1f} | {win_lbl:<12}")

            current_time = start_service

            # 4. Check Repeated Visits
            if route[i] != 0:
                if route[i] in visited_customers:
                    return False, f"Node {route[i]} visited multiple times."
                visited_customers.add(route[i])

        total_cost += route_dist
        if verbose: print(f"  => Route Summary: Dist={route_dist:.2f}, Load={current_load}")

    # 5. Check Missing Customers
    num_nodes = len(customers)
    real_customers = set(range(1, num_nodes - 1))
    missing = real_customers - visited_customers

    if missing:
        msg = f"Missing customers: {sorted(list(missing))}"
        if verbose: print(f"\n>>> FAILED: {msg}")
        return False, msg

    if verbose:
        print("-" * 80)
        print(f"VALIDATION SUCCESS. Total Cost: {total_cost:.2f}")
        print("=" * 80 + "\n")

    return True, "VALID"

def extract_routes(
    x: Dict[Any, Any],
    customers: List[Customer],
    vehicle: VehicleInfo,
    solver: pywraplp.Solver | cp_model.CpSolver | None
) -> List[List[int]]:
    """Trích xuất lộ trình, hỗ trợ đa nền tảng solver."""
    if not x: return []
    routes = []
    N = len(customers)

    # Helper: Kiểm tra biến được chọn hay không dựa trên loại solver
    def is_selected(var):
        # CP-SAT Logic
        if isinstance(solver, cp_model.CpSolver):
            return solver.Value(var) == 1
        # MILP Logic
        if hasattr(var, 'solution_value'):
            return var.solution_value() > 0.5
        return False

    # Check Dimension
    sample_key = next(iter(x))
    is_3d = len(sample_key) == 3

    if is_3d:
        for k in range(vehicle.number):
            route = [0]
            curr = 0
            while True:
                found_next = False
                for j in range(N):
                    if curr == j: continue
                    if (curr, j, k) in x and is_selected(x[curr, j, k]):
                        route.append(j)
                        curr = j
                        found_next = True
                        break
                if curr == N - 1 or not found_next: break

            if len(route) > 1:
                if route[-1] == N - 1: route[-1] = 0
                routes.append(route)
    else:
        # 2D Logic
        start_nodes = [j for j in range(1, N) if (0, j) in x and is_selected(x[0, j])]
        for start_node in start_nodes:
            route = [0, start_node]
            curr = start_node
            while True:
                if curr == N - 1: break
                found_next = False
                for j in range(N):
                    if curr == j: continue
                    if (curr, j) in x and is_selected(x[curr, j]):
                        route.append(j)
                        curr = j
                        found_next = True
                        break
                if not found_next: break

            if route[-1] == N - 1: route[-1] = 0
            routes.append(route)

    return routes

# --- 3. CORE PROCESSING ---

def process_single_instance(
    file_path: Path,
    config: BenchmarkConfig,
    solver_func: SolverCallable
) -> BenchmarkResult:

    file_name = file_path.name
    try:
        name, vehicle, customers = read_solomon_vrptw(str(file_path))
        start_time = time.perf_counter()

        # RUN SOLVER
        solver, status, x, new_customers = solver_func(
            customers,
            vehicle,
            config.limit_nodes,
            config.time_limit_sec
        )

        duration = time.perf_counter() - start_time

        # --- UNIFY STATUS CODES ---
        # Gom nhóm các trạng thái thành công của cả 2 thư viện
        SUCCESS_STATUSES = [
            pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE,
            cp_model.OPTIMAL, cp_model.FEASIBLE
        ]

        routes = []
        obj_val = "N/A"
        status_str = "UNKNOWN"
        is_valid = False
        val_msg = "Not Solved"

        if status in SUCCESS_STATUSES:
            # 1. Determine Status String
            if status in [pywraplp.Solver.OPTIMAL, cp_model.OPTIMAL]:
                status_str = "OPTIMAL"
            else:
                status_str = "FEASIBLE"

            # 2. Get Objective Value (Handle differences)
            if isinstance(solver, cp_model.CpSolver):
                obj_val = f"{solver.ObjectiveValue():.2f}"
            elif hasattr(solver, 'Objective'):
                obj_val = f"{solver.Objective().Value():.2f}"

            # 3. Extract & Validate
            routes = extract_routes(x, new_customers, vehicle, solver)
            is_valid, val_msg = validate_solution(routes, new_customers, vehicle, config.verbose)

        else:
            # Handle Failure Codes
            if status == cp_model.INFEASIBLE or status == pywraplp.Solver.INFEASIBLE:
                status_str = "INFEASIBLE"
            elif status == cp_model.MODEL_INVALID:
                status_str = "INVALID"
            else:
                status_str = "NO_SOL/TIMEOUT"

        return BenchmarkResult(file_name, status_str, duration, obj_val, routes, is_valid, val_msg)

    except Exception as e:
        return BenchmarkResult(file_name, "ERROR", 0.0, "N/A", [], False, str(e))

# --- 4. MAIN ---

def main():
    # --- A. CẤU HÌNH ---

    # 1. Chọn hàm solver bạn muốn dùng ở đây!
    CURRENT_SOLVER = solve_cvrptw_milp_gemini_scip

    config = BenchmarkConfig(
        input_folder=Path("testcases"),
        output_dir=Path("milp_results"),
        output_name="gemini_scip_100.txt", # Đặt tên file output
        limit_nodes=100,
        time_limit_sec=100,
        verbose=True
    )

    # --- B. CHUẨN BỊ MÔI TRƯỜNG ---
    if not config.output_dir.exists():
        print(f"Creating directory: {config.output_dir}")
        config.output_dir.mkdir(parents=True, exist_ok=True)

    full_output_path = config.output_dir / config.output_name
    test_files = sorted(list(config.input_folder.glob("*.txt")))

    if not test_files:
        print(f"Warning: No input files found in {config.input_folder}")
        return

    print(f"Solver: {CURRENT_SOLVER.__name__}")
    print(f"Found {len(test_files)} files. Output: {full_output_path}")

    headers = [f"{'Instance':<15}", f"{'Status':<10}", f"{'Time(s)':<8}", f"{'Obj Val':<10}", f"{'Valid':<6}", f"{'Routes / Error'}"]

    # --- C. CHẠY VÒNG LẶP ---
    with open(full_output_path, "w", encoding="utf-8") as f:
        f.write(" | ".join(headers) + "\n" + "-" * 120 + "\n")
        print(" | ".join(headers))

        total_start = time.perf_counter()

        for file_path in test_files:
            # Truyền CURRENT_SOLVER vào hàm xử lý
            res = process_single_instance(file_path, config, solver_func=CURRENT_SOLVER)

            # Format output
            routes_str = str(res.routes) if res.status != "ERROR" else res.validation_msg
            display_routes = (routes_str[:50] + '...') if len(routes_str) > 50 else routes_str
            valid_mark = "OK" if res.is_valid else "FAIL"
            if res.status == "NO_SOL/TIMEOUT": valid_mark = "-"

            line = f"{res.instance_name:<15} | {res.status:<10} | {res.duration:<8.4f} | {res.obj_value:<10} | {valid_mark:<6} | {display_routes}"

            f.write(line + "\n")
            if not res.is_valid and res.status not in ["NO_SOL/TIMEOUT", "ERROR"]:
                 f.write(f"   >>> Error: {res.validation_msg}\n")

            print(line)
        summary = f"\nDone! Total time: {time.perf_counter() - total_start:.2f}s"
        f.write(summary)
        print(summary)
        f.flush()


if __name__ == "__main__":
    main()