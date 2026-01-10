import time
import math
import os
import glob
from typing import List, Tuple, Dict, Optional
from ortools.linear_solver import pywraplp

# Import các class data từ file parse của bạn
# Đảm bảo file parse.py nằm cùng thư mục
from parse import read_solomon_vrptw, VehicleInfo, Customer

# --- Type Aliases ---
SolverVar = pywraplp.Variable
RouteKey = Tuple[int, int, int]
NodeVehicleKey = Tuple[int, int]

def euclidean(a: Customer, b: Customer) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def manhattan(a: Customer, b: Customer) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)

def build_distance(customers: List[Customer]) -> List[List[int]]:
    n = len(customers)
    dist: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = manhattan(customers[i], customers[j])
    return dist

def solve_cvrptw_milp_sat(
    original_customers: List[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> Tuple[Optional[pywraplp.Solver], int, Dict[RouteKey, SolverVar], List[Customer]]:
    customers = original_customers[:limit_nodes]
    start = customers[0]
    end = Customer(
        id=len(customers),
        x=start.x,
        y=start.y,
        demand=0,
        ready_time=start.ready_time,
        due_date=start.due_date,
        service_time=0
    )
    customers.append(end)

    N = len(customers)
    K = vehicle.number
    Q = vehicle.capacity
    dist = build_distance(customers)
    BIG_M = 2_000

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SAT")
    if not solver:
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers
    solver.SetTimeLimit(time_limit_sec * 1000)

    x: dict[RouteKey, SolverVar] = {}
    t: dict[NodeVehicleKey, SolverVar] = {}
    for k in range(K):
        for i in range(N):
            t[i, k] = solver.IntVar(customers[i].ready_time, customers[i].due_date, f't[{i}, {k}]')
            for j in range(N):
                if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                    continue
                x[i, j, k] = solver.BoolVar(f'x[{i}, {j}, {k}]')

    solver.Minimize(
        solver.Sum(
            dist[i][j] * x[i, j, k]
            for k in range(K)
            for i in range(N)
            for j in range(N)
            if (i, j, k) in x
        )
    )

    for i in range(1, N - 1):
        solver.Add(
            solver.Sum(
                x[j, i, k]
                for k in range(K)
                for j in range(0, N - 1)
                if i != j
            ) == 1
        )

    for (i, j, k) in x:
        solver.Add(t[j, k] >= t[i, k] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j, k]))

    for k in range(K):
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            ) <= 1
        )
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            )
            ==
            solver.Sum(
                x[i, N - 1, k]
                for i in range(1, N - 1)
            )
        )
    for k in range(K):
        for i in range(1, N - 1):
            solver.Add(
                solver.Sum(
                    x[i, j, k]
                    for j in range(1, N)
                    if i != j
                )
                ==
                solver.Sum(
                    x[j, i, k]
                    for j in range(0, N - 1)
                    if i != j
                )
            )
    for k in range(K):
            solver.Add(
                solver.Sum(
                    customers[i].demand
                    *
                    solver.Sum(
                        x[j, i, k]
                        for j in range(0, N - 1)
                        if i != j
                    )
                    for i in range(1, N - 1)
                ) <= Q
            )
    status = solver.Solve()
    return solver, status, x, customers

def solve_cvrptw_milp_sat_2d(
    original_customers: List[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> Tuple[Optional[pywraplp.Solver], int, Dict[tuple[int, int], SolverVar], List[Customer]]:
    customers = original_customers[:limit_nodes]
    start = customers[0]
    end = Customer(
        id=len(customers),
        x=start.x,
        y=start.y,
        demand=0,
        ready_time=start.ready_time,
        due_date=start.due_date,
        service_time=0
    )
    customers.append(end)

    N = len(customers)
    K = vehicle.number
    Q = vehicle.capacity
    dist = build_distance(customers)
    BIG_M = 2_000

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SAT")
    if not solver:
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers
    solver.SetTimeLimit(time_limit_sec * 1000)

    x: dict[tuple[int, int], SolverVar] = {}
    t: dict[int, SolverVar] = {}
    u: dict[int, SolverVar] = {}
    for i in range(N):
        t[i] = solver.IntVar(customers[i].ready_time, customers[i].due_date, f't[{i}]')
        u[i] = solver.IntVar(customers[i].demand, Q if i != 0 else 0, f'u[{i}]')
        for j in range(N):
            if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                continue
            x[i, j] = solver.BoolVar(f'x[{i}, {j}]')

    solver.Minimize(
        solver.Sum(
            dist[i][j] * x[i, j]
            for i in range(N)
            for j in range(N)
            if (i, j) in x
        )
    )

    for i in range(1, N - 1):
        solver.Add(
            solver.Sum(
                x[j, i]
                for j in range(0, N - 1)
                if i != j
            ) == 1
        )

    for (i, j) in x:
        solver.Add(t[j] >= t[i] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j]))

    for (i, j) in x:
        solver.Add(u[j] >= u[i] + customers[j].demand - Q * (1 - x[i, j]))

    solver.Add(
        solver.Sum(
            x[0, i]
            for i in range(1, N - 1)
        ) <= vehicle.number
    )
    solver.Add(
        solver.Sum(
            x[0, i]
            for i in range(1, N - 1)
        )
        ==
        solver.Sum(
            x[i, N - 1]
            for i in range(1, N - 1)
        )
    )

    for i in range(1, N - 1):
        solver.Add(
            solver.Sum(
                x[i, j]
                for j in range(1, N)
                if i != j
            )
            ==
            solver.Sum(
                x[j, i]
                for j in range(0, N - 1)
                if i != j
            )
        )
    status = solver.Solve()
    return solver, status, x, customers

def solve_cvrptw_milp_sat_with_load_var(
    original_customers: List[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> Tuple[Optional[pywraplp.Solver], int, Dict[RouteKey, SolverVar], List[Customer]]:
    customers = original_customers[:limit_nodes]
    start = customers[0]
    end = Customer(
        id=len(customers),
        x=start.x,
        y=start.y,
        demand=0,
        ready_time=start.ready_time,
        due_date=start.due_date,
        service_time=0
    )
    customers.append(end)

    N = len(customers)
    K = vehicle.number
    Q = vehicle.capacity
    dist = build_distance(customers)
    BIG_M = 2_000

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SAT")
    if not solver:
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers
    solver.SetTimeLimit(time_limit_sec * 1000)

    x: dict[RouteKey, SolverVar] = {}
    t: dict[NodeVehicleKey, SolverVar] = {}
    u: dict[NodeVehicleKey, SolverVar] = {}
    for k in range(K):
        for i in range(N):
            t[i, k] = solver.IntVar(customers[i].ready_time, customers[i].due_date, f't[{i}, {k}]')
            u[i, k] = solver.IntVar(customers[i].demand, Q if i != 0 else 0, f'u[{i}, {k}]')
            for j in range(N):
                if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                    continue
                x[i, j, k] = solver.BoolVar(f'x[{i}, {j}, {k}]')

    solver.Minimize(
        solver.Sum(
            dist[i][j] * x[i, j, k]
            for k in range(K)
            for i in range(N)
            for j in range(N)
            if (i, j, k) in x
        )
    )

    for i in range(1, N - 1):
        solver.Add(
            solver.Sum(
                x[j, i, k]
                for k in range(K)
                for j in range(0, N - 1)
                if i != j
            ) == 1
        )

    for (i, j, k) in x:
        solver.Add(t[j, k] >= t[i, k] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j, k]))

    for (i, j, k) in x:
        solver.Add(u[j, k] >= u[i, k] + customers[j].demand - Q * (1 - x[i, j, k]))

    for k in range(K):
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            ) <= 1
        )
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            )
            ==
            solver.Sum(
                x[i, N - 1, k]
                for i in range(1, N - 1)
            )
        )
    for k in range(K):
        for i in range(1, N - 1):
            solver.Add(
                solver.Sum(
                    x[i, j, k]
                    for j in range(1, N)
                    if i != j
                )
                ==
                solver.Sum(
                    x[j, i, k]
                    for j in range(0, N - 1)
                    if i != j
                )
            )
    status = solver.Solve()
    return solver, status, x, customers

def solve_cvrptw_milp_scip_with_load_var(
    original_customers: List[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> Tuple[Optional[pywraplp.Solver], int, Dict[RouteKey, SolverVar], List[Customer]]:
    customers = original_customers[:limit_nodes]
    start = customers[0]
    end = Customer(
        id=len(customers),
        x=start.x,
        y=start.y,
        demand=0,
        ready_time=start.ready_time,
        due_date=start.due_date,
        service_time=0
    )
    customers.append(end)

    N = len(customers)
    K = vehicle.number
    Q = vehicle.capacity
    dist = build_distance(customers)
    BIG_M = 2_000

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers
    solver.SetTimeLimit(time_limit_sec * 1000)

    x: dict[RouteKey, SolverVar] = {}
    t: dict[NodeVehicleKey, SolverVar] = {}
    u: dict[NodeVehicleKey, SolverVar] = {}
    for k in range(K):
        for i in range(N):
            t[i, k] = solver.NumVar(customers[i].ready_time, customers[i].due_date, f't[{i}, {k}]')
            u[i, k] = solver.NumVar(0, Q if i != 0 else 0, f'u[{i}, {k}]')
            for j in range(N):
                if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                    continue
                x[i, j, k] = solver.BoolVar(f'x[{i}, {j}, {k}]')

    solver.Minimize(
        solver.Sum(
            dist[i][j] * x[i, j, k]
            for k in range(K)
            for i in range(N)
            for j in range(N)
            if (i, j, k) in x
        )
    )

    for i in range(1, N - 1):
        solver.Add(
            solver.Sum(
                x[j, i, k]
                for k in range(K)
                for j in range(0, N - 1)
                if i != j
            ) == 1
        )

    for (i, j, k) in x:
        solver.Add(t[j, k] >= t[i, k] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j, k]))

    for (i, j, k) in x:
        solver.Add(u[j, k] >= u[i, k] + customers[j].demand - Q * (1 - x[i, j, k]))

    for k in range(K):
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            ) <= 1
        )
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            )
            ==
            solver.Sum(
                x[i, N - 1, k]
                for i in range(1, N - 1)
            )
        )
    for k in range(K):
        for i in range(1, N - 1):
            solver.Add(
                solver.Sum(
                    x[i, j, k]
                    for j in range(1, N)
                    if i != j
                )
                ==
                solver.Sum(
                    x[j, i, k]
                    for j in range(0, N - 1)
                    if i != j
                )
            )
    for k in range(K):
            solver.Add(
                solver.Sum(
                    customers[i].demand
                    *
                    solver.Sum(
                        x[j, i, k]
                        for j in range(0, N - 1)
                        if i != j
                    )
                    for i in range(1, N - 1)
                ) <= Q
            )
    status = solver.Solve()
    return solver, status, x, customers

def solve_cvrptw_milp_my_scip(
    original_customers: List[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> Tuple[Optional[pywraplp.Solver], int, Dict[RouteKey, SolverVar], List[Customer]]:
    customers = original_customers[:limit_nodes]
    start = customers[0]
    end = Customer(
        id=len(customers),
        x=start.x,
        y=start.y,
        demand=0,
        ready_time=start.ready_time,
        due_date=start.due_date,
        service_time=0
    )
    customers.append(end)

    N = len(customers)
    K = vehicle.number
    Q = vehicle.capacity
    dist = build_distance(customers)
    BIG_M = 2_000

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers
    solver.SetTimeLimit(time_limit_sec * 1000)

    x: dict[RouteKey, SolverVar] = {}
    t: dict[NodeVehicleKey, SolverVar] = {}
    for k in range(K):
        for i in range(N):
            t[i, k] = solver.NumVar(customers[i].ready_time, customers[i].due_date, f't[{i}, {k}]')
            for j in range(N):
                if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                    continue
                x[i, j, k] = solver.BoolVar(f'x[{i}, {j}, {k}]')

    solver.Minimize(
        solver.Sum(
            dist[i][j] * x[i, j, k]
            for k in range(K)
            for i in range(N)
            for j in range(N)
            if (i, j, k) in x
        )
    )

    for i in range(1, N - 1):
        solver.Add(
            solver.Sum(
                x[j, i, k]
                for k in range(K)
                for j in range(0, N - 1)
                if i != j
            ) == 1
        )

    for (i, j, k) in x:
        solver.Add(t[j, k] >= t[i, k] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j, k]))

    for k in range(K):
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            ) <= 1
        )
        solver.Add(
            solver.Sum(
                x[0, i, k]
                for i in range(1, N - 1)
            )
            ==
            solver.Sum(
                x[i, N - 1, k]
                for i in range(1, N - 1)
            )
        )
    for k in range(K):
        for i in range(1, N - 1):
            solver.Add(
                solver.Sum(
                    x[i, j, k]
                    for j in range(1, N)
                    if i != j
                )
                ==
                solver.Sum(
                    x[j, i, k]
                    for j in range(0, N - 1)
                    if i != j
                )
            )
    for k in range(K):
            solver.Add(
                solver.Sum(
                    customers[i].demand
                    *
                    solver.Sum(
                        x[j, i, k]
                        for j in range(0, N - 1)
                        if i != j
                    )
                    for i in range(1, N - 1)
                ) <= Q
            )
    status = solver.Solve()
    return solver, status, x, customers

def solve_cvrptw_milp_scip(
    original_customers: List[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,     # Giảm xuống 25 để demo nhanh, tăng lên 50 hoặc 100 nếu muốn chạy thật
    time_limit_sec: int = 30   # Giới hạn thời gian mỗi test case (giây)
) -> Tuple[Optional[pywraplp.Solver], int, Dict[RouteKey, SolverVar], List[Customer]]:

    # 1. Cắt giảm dữ liệu
    customers: List[Customer] = original_customers[:limit_nodes]

    # 2. Tạo Dummy Depot (Node n-1)
    depot_start = customers[0]
    depot_end = Customer(
        id=len(customers),
        x=depot_start.x,
        y=depot_start.y,
        demand=0,
        ready_time=depot_start.ready_time,
        due_date=depot_start.due_date,
        service_time=0
    )
    customers.append(depot_end)

    n = len(customers)
    K = vehicle.number
    Q = vehicle.capacity
    dist = build_distance(customers)

    # Sử dụng SCIP backend
    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers

    # --- CẤU HÌNH QUAN TRỌNG: Time Limit ---
    # Đặt giới hạn thời gian (ms) để solver không bị treo nếu bài toán quá khó
    solver.SetTimeLimit(time_limit_sec * 1000)

    BIG_M = 100000

    # ---- Variables ----
    x: Dict[RouteKey, SolverVar] = {}
    t: Dict[NodeVehicleKey, SolverVar] = {}
    u: Dict[NodeVehicleKey, SolverVar] = {}

    for k in range(K):
        for i in range(n):
            t[i, k] = solver.NumVar(float(customers[i].ready_time), float(customers[i].due_date), f"t[{i},{k}]")
            u[i, k] = solver.NumVar(0.0, float(Q), f"u[{i},{k}]")

            for j in range(n):
                if i == j: continue
                if j == 0: continue
                if i == n - 1: continue

                x[i, j, k] = solver.BoolVar(f"x[{i},{j},{k}]")

    # ---- Objective ----
    solver.Minimize(
        solver.Sum(dist[i][j] * x[i, j, k]
                   for k in range(K)
                   for i in range(n)
                   for j in range(n)
                   if (i, j, k) in x)
    )

    # ---- Constraints ----
    # 1. Visit Constraints
    for i in range(1, n - 1):
        solver.Add(solver.Sum(x[i, j, k] for k in range(K) for j in range(n) if (i, j, k) in x) == 1)

    # 2. Flow Conservation
    for k in range(K):
        for h in range(1, n - 1):
            sum_in = solver.Sum(x[i, h, k] for i in range(n) if (i, h, k) in x)
            sum_out = solver.Sum(x[h, j, k] for j in range(n) if (h, j, k) in x)
            solver.Add(sum_in == sum_out)

    # 3. Depot Constraints
    for k in range(K):
        solver.Add(solver.Sum(x[0, j, k] for j in range(1, n) if (0, j, k) in x) <= 1)
        solver.Add(solver.Sum(x[i, n-1, k] for i in range(n-1) if (i, n-1, k) in x) <= 1)

        sum_start = solver.Sum(x[0, j, k] for j in range(1, n) if (0, j, k) in x)
        sum_end = solver.Sum(x[i, n-1, k] for i in range(n-1) if (i, n-1, k) in x)
        solver.Add(sum_start == sum_end)

    # 4. Time Windows (MTZ)
    for k in range(K):
        for i in range(n):
            for j in range(n):
                if (i, j, k) in x:
                    solver.Add(t[j, k] >= t[i, k] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j, k]))

    # 5. Capacity
    for k in range(K):
        solver.Add(u[0, k] == 0)
        for i in range(n):
            for j in range(n):
                if (i, j, k) in x:
                    solver.Add(u[j, k] >= u[i, k] + customers[j].demand - Q * (1 - x[i, j, k]))

    status: int = solver.Solve()
    return solver, status, x, customers

def validate_solution(routes: List[List[int]], customers: List[Customer], vehicle: VehicleInfo) -> Tuple[bool, str]:
    visited_customers = set()
    print("\n--- START VALIDATION (MANHATTAN 2D/3D) ---")

    for r_idx, route in enumerate(routes):
        if not route: continue
        
        if route[0] != 0 or route[-1] != 0:
            return False, f"Route {r_idx} struct error: {route}"

        current_load = 0
        current_time = float(customers[0].ready_time)

        print(f"Vehicle {r_idx}: [0] Start Time={current_time}")

        for i in range(1, len(route)):
            prev_node_idx = route[i-1]
            curr_node_idx = route[i]

            prev_cust = customers[prev_node_idx]
            curr_cust = customers[curr_node_idx]

            current_load += curr_cust.demand
            if current_load > vehicle.capacity:
                return False, f"Route {r_idx}: Overload at {curr_node_idx} ({current_load}/{vehicle.capacity})"

            travel_time = manhattan(prev_cust, curr_cust)
            arrival_time = current_time + prev_cust.service_time + travel_time

            if arrival_time > curr_cust.due_date:
                return False, f"Route {r_idx}: Late at {curr_node_idx} (Arr: {arrival_time} > Due: {curr_cust.due_date})"

            start_service = max(arrival_time, curr_cust.ready_time)

            print(f"  -> {curr_node_idx}: Load({current_load}) | Dist({travel_time}) | Arrive({arrival_time}) | Wait({max(0, curr_cust.ready_time - arrival_time)}) | Window[{curr_cust.ready_time}-{curr_cust.due_date}]")

            current_time = start_service

            if curr_node_idx != 0:
                if curr_node_idx in visited_customers:
                    return False, f"Node {curr_node_idx} repeated."
                visited_customers.add(curr_node_idx)

    num_real_nodes = len(customers) - 1
    all_real_customers = set(range(1, num_real_nodes))
    
    missing = all_real_customers - visited_customers
    if missing:
        return False, f"Missing customers: {missing}"

    return True, "VALID"

def extract_routes(x: Dict[Tuple[int, ...], SolverVar], customers: List[Customer], vehicle: ValueError) -> List[List[int]]:
    routes = []
    N = len(customers)
    
    if not x:
        return []

    first_key = next(iter(x))
    is_3d = len(first_key) == 3

    if is_3d:
        for k in range(vehicle.number):
            route = []
            curr = 0
            while True:
                route.append(curr)
                if curr == N - 1: break

                found = False
                for j in range(N):
                    if (curr, j, k) in x and x[curr, j, k].solution_value() > 0.5:
                        curr = j
                        found = True
                        break
                if not found: break

            if len(route) > 1 and route[-1] == N - 1:
                clean_route = [node if node != N - 1 else 0 for node in route]
                routes.append(clean_route)
    else:
        start_nodes = []
        for j in range(1, N):
            if (0, j) in x and x[0, j].solution_value() > 0.5:
                start_nodes.append(j)
        
        for start_node in start_nodes:
            route = [0]
            curr = start_node
            
            while True:
                route.append(curr)
                if curr == N - 1: break
                
                found_next = False
                for j in range(N):
                    if (curr, j) in x and x[curr, j].solution_value() > 0.5:
                        curr = j
                        found_next = True
                        break
                if not found_next: break
            
            if route[-1] == N - 1:
                route[-1] = 0
            routes.append(route)

    is_valid, msg = validate_solution(routes, customers, vehicle)
    if is_valid:
        print(f">>> VALIDATION SUCCESS")
    else:
        print(f">>> VALIDATION FAILED: {msg}")

    return routes

def main() -> None:
    # 1. Cấu hình thư mục và file output
    input_folder = "testcases"
    output_file = "milp_sat_2d_50_results.txt"

    # Tìm tất cả file .txt trong folder
    test_files = sorted(glob.glob(os.path.join(input_folder, "*.txt")))

    if not test_files:
        print(f"Warning: Không tìm thấy file .txt nào trong thư mục '{input_folder}'")
        return

    print(f"Found {len(test_files)} test files. Starting benchmark...\n")

    # Mở file để ghi kết quả (mode 'w' sẽ ghi đè file cũ)
    with open(output_file, "w", encoding="utf-8") as f_out:

        # Ghi header vào file
        header = f"{'Instance':<15} | {'Status':<15} | {'Time(s)':<10} | {'Obj Value':<15} | {'Routes'}"
        f_out.write(header + "\n")
        f_out.write("-" * 100 + "\n")
        print(header) # In ra màn hình console để theo dõi

        total_start = time.perf_counter()

        for file_path in test_files:
            file_name = os.path.basename(file_path)

            try:
                # Đọc dữ liệu
                name, vehicle, customers = read_solomon_vrptw(file_path)

                # Bắt đầu đo giờ cho từng file
                start_time = time.perf_counter()

                # --- CHẠY SOLVER ---
                # Lưu ý: limit_nodes=25 để chạy nhanh test.
                # MILP rất chậm, nếu để 50 hoặc 100 có thể mất hàng giờ.
                solver, status, x, new_customers = solve_cvrptw_milp_sat_2d(
                    customers,
                    vehicle,
                    limit_nodes=50,
                    time_limit_sec=30
                )

                duration = time.perf_counter() - start_time

                # Xử lý kết quả
                status_str = "UNKNOWN"
                obj_val = "N/A"
                routes_str = "[]"

                if status == pywraplp.Solver.OPTIMAL:
                    status_str = "OPTIMAL"
                    obj_val = f"{solver.Objective().Value():.2f}"
                    routes = extract_routes(x, new_customers, vehicle)
                    routes_str = str(routes)
                elif status == pywraplp.Solver.FEASIBLE:
                    status_str = "FEASIBLE"
                    obj_val = f"{solver.Objective().Value():.2f}"
                    routes = extract_routes(x, new_customers, vehicle)
                    routes_str = str(routes)
                else:
                    status_str = "NO_SOL/TIMEOUT"

                # Format dòng kết quả
                result_line = f"{file_name:<15} | {status_str:<15} | {duration:<10.4f} | {obj_val:<15} | {routes_str}"

                # Ghi vào file và in ra màn hình
                f_out.write(result_line + "\n")
                f_out.flush() # Đảm bảo ghi xuống đĩa ngay lập tức
                print(result_line)

            except Exception as e:
                error_msg = f"{file_name:<15} | ERROR: {str(e)}"
                f_out.write(error_msg + "\n")
                print(error_msg)
                raise

        total_duration = time.perf_counter() - total_start
        summary = f"\nDone! Total benchmark time: {total_duration:.2f} seconds. Results saved to {output_file}"
        f_out.write(summary)
        print(summary)

if __name__ == "__main__":
    main()