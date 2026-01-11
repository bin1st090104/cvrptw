from milp.gemini_scip import solve_cvrptw_milp_gemini_scip
from milp.scip import solve_cvrptw_milp_scip
from milp.scip_with_load_vars import solve_cvrptw_milp_scip_with_load_vars
from milp.sat_with_load_vars import solve_cvrptw_milp_sat_with_load_vars
from milp.sat_2d import solve_cvrptw_milp_sat_2d
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
    Tuple[pywraplp.Solver | None, int, Dict[Any, SolverVar], List[Customer]]
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
    """
    Kiểm tra tính hợp lệ và in chi tiết lộ trình nếu verbose=True.
    Không thay đổi logic kiểm tra, chỉ thêm phần hiển thị (logging).
    """
    visited_customers = set()
    total_cost = 0.0

    if verbose:
        print("\n" + "="*80)
        print(f"{'VALIDATION DETAIL REPORT':^80}")
        print("="*80)

    for r_idx, route in enumerate(routes):
        if not route: continue

        # --- VALIDATE 1: Cấu trúc route ---
        if route[0] != 0 or route[-1] != 0:
            return False, f"Route {r_idx} Error: Route must start and end at Depot (0). Got: {route}"

        # Khởi tạo trạng thái xe
        current_load = 0
        # Thời gian bắt đầu tính từ ready_time của Depot
        current_time = customers[0].ready_time
        route_dist = 0.0

        if verbose:
            print(f"\nVehicle {r_idx + 1} (Cap: {vehicle.capacity}): {route}")
            print(f"  {'Node':<5} | {'Load':<8} | {'Travel':<8} | {'Arrive':<8} | {'Wait':<6} | {'Start':<8} | {'Window':<12}")
            print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*12}")
            # In dòng khởi tạo tại Depot
            print(f"  {'0(D)':<5} | {'0':<8} | {'0.0':<8} | {'-':<8} | {'-':<6} | {current_time:<8.1f} | [{customers[0].ready_time}-{customers[0].due_date}]")

        for i in range(1, len(route)):
            prev_idx = route[i-1]
            curr_idx = route[i]

            prev_cust = customers[prev_idx]
            curr_cust = customers[curr_idx]

            # --- VALIDATE 2: Tải trọng ---
            current_load += curr_cust.demand
            if current_load > vehicle.capacity:
                msg = f"Route {r_idx} FAILED: Overload at Node {curr_idx}. Load {current_load} > Cap {vehicle.capacity}."
                if verbose: print(f"  >>> {msg}")
                return False, msg

            # --- VALIDATE 3: Thời gian ---
            # Thời gian di chuyển
            dist = manhattan(prev_cust, curr_cust)
            route_dist += dist

            # Thời gian đến = (Start prev) + (Service prev) + (Travel)
            arrival_time = current_time + prev_cust.service_time + dist

            # Thời gian chờ (nếu đến sớm hơn ready_time)
            wait_time = max(0, curr_cust.ready_time - arrival_time)

            # Thời gian bắt đầu phục vụ thực tế
            start_service = max(arrival_time, curr_cust.ready_time)

            # Check Due Date
            if arrival_time > curr_cust.due_date:
                msg = f"Route {r_idx} FAILED: Late at Node {curr_idx}. Arrive {arrival_time:.1f} > Due {curr_cust.due_date}."
                if verbose: print(f"  >>> {msg}")
                return False, msg

            if verbose:
                # In thông tin từng bước di chuyển
                node_lbl = str(curr_idx) if curr_idx != 0 else "0(D)"
                win_lbl = f"[{curr_cust.ready_time}-{curr_cust.due_date}]"
                print(f"  {node_lbl:<5} | {current_load:<3}/{vehicle.capacity:<4} | {dist:<8.1f} | {arrival_time:<8.1f} | {wait_time:<6.1f} | {start_service:<8.1f} | {win_lbl:<12}")

            # Cập nhật thời gian hiện tại = Thời gian bắt đầu phục vụ
            current_time = start_service

            # --- VALIDATE 4: Thăm lặp ---
            if curr_idx != 0:
                if curr_idx in visited_customers:
                    return False, f"Node {curr_idx} visited multiple times."
                visited_customers.add(curr_idx)

        total_cost += route_dist
        if verbose:
            print(f"  => Route {r_idx + 1} Summary: Load={current_load}/{vehicle.capacity}, Dist={route_dist:.2f}")

    # --- VALIDATE 5: Bỏ sót khách ---
    num_nodes = len(customers)
    # Khách hàng thực là 1..N-2 (Do N-1 là Dummy Depot)
    real_customers = set(range(1, num_nodes - 1))
    missing = real_customers - visited_customers

    if missing:
        msg = f"FAILED: Missing customers: {sorted(list(missing))}"
        if verbose: print(f"\n>>> {msg}")
        return False, msg

    if verbose:
        print("-" * 80)
        print(f"VALIDATION SUCCESS. Total Cost (Dist): {total_cost:.2f}")
        print("=" * 80 + "\n")

    return True, "VALID"

def extract_routes(
    x: Dict[Any, SolverVar],
    customers: List[Customer],
    vehicle: VehicleInfo
) -> List[List[int]]:
    """
    Trích xuất lộ trình từ biến x. Hỗ trợ tự động nhận diện 2D hoặc 3D.
    """
    if not x: return []
    routes = []
    N = len(customers)

    # Check dimension: Lấy 1 key bất kỳ để xem nó có mấy phần tử
    sample_key = next(iter(x))
    is_3d = len(sample_key) == 3

    if is_3d:
        # --- Logic cho biến 3D x[i, j, k] ---
        for k in range(vehicle.number):
            route = [0]
            curr = 0
            while True:
                found_next = False
                for j in range(N):
                    if curr == j: continue
                    if (curr, j, k) in x and x[curr, j, k].solution_value() > 0.5:
                        route.append(j)
                        curr = j
                        found_next = True
                        break
                if curr == N - 1 or not found_next: break

            if len(route) > 1:
                if route[-1] == N - 1: route[-1] = 0
                routes.append(route)
    else:
        # --- Logic cho biến 2D x[i, j] ---
        # 1. Tìm tất cả các chuyến xuất phát từ 0
        start_nodes = [j for j in range(1, N) if (0, j) in x and x[0, j].solution_value() > 0.5]

        # 2. Trace từng chuyến
        for start_node in start_nodes:
            route = [0, start_node]
            curr = start_node
            while True:
                if curr == N - 1: break
                found_next = False
                for j in range(N):
                    if curr == j: continue
                    if (curr, j) in x and x[curr, j].solution_value() > 0.5:
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
    solver_func: SolverCallable  # <-- Dependency Injection ở đây
) -> BenchmarkResult:

    file_name = file_path.name
    try:
        name, vehicle, customers = read_solomon_vrptw(str(file_path))
        start_time = time.perf_counter()

        # Gọi hàm solver được truyền vào
        solver, status, x, new_customers = solver_func(
            customers,
            vehicle,
            config.limit_nodes,
            config.time_limit_sec
        )

        duration = time.perf_counter() - start_time

        # Xử lý kết quả
        routes = []
        obj_val = "N/A"
        status_str = "UNKNOWN"
        is_valid = False
        val_msg = "Not Solved"

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
            obj_val = f"{solver.Objective().Value():.2f}"

            routes = extract_routes(x, new_customers, vehicle)
            is_valid, val_msg = validate_solution(routes, new_customers, vehicle, config.verbose)
        else:
            status_str = "NO_SOL/TIMEOUT"

        return BenchmarkResult(file_name, status_str, duration, obj_val, routes, is_valid, val_msg)

    except Exception as e:
        return BenchmarkResult(file_name, "ERROR", 0.0, "N/A", [], False, str(e))

# --- 4. MAIN ---

def main():
    # --- A. CẤU HÌNH ---

    # 1. Chọn hàm solver bạn muốn dùng ở đây!
    # CURRENT_SOLVER = solve_cvrptw_milp_sat      # Dùng cái này cho 3D
    CURRENT_SOLVER = solve_cvrptw_milp_sat_2d  # Dùng cái này cho 2D

    config = BenchmarkConfig(
        input_folder=Path("testcases"),
        output_dir=Path("milp_results"),
        output_name="sat_2d_15.txt", # Đặt tên file output
        limit_nodes=15,
        time_limit_sec=30,
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
            valid_mark = "OK" if res.is_valid else "FAIL"
            if res.status == "NO_SOL/TIMEOUT": valid_mark = "-"

            line = f"{res.instance_name:<15} | {res.status:<10} | {res.duration:<8.4f} | {res.obj_value:<10} | {valid_mark:<6} | {routes_str}"

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