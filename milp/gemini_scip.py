from .utils import RouteKey, SolverVar, build_distance, NodeVehicleKey
from ortools.linear_solver import pywraplp
from parse import VehicleInfo, Customer


def solve_cvrptw_milp_gemini_scip(
    original_customers: list[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,     # Giảm xuống 25 để demo nhanh, tăng lên 50 hoặc 100 nếu muốn chạy thật
    time_limit_sec: int = 30   # Giới hạn thời gian mỗi test case (giây)
) -> tuple[pywraplp.Solver | None, int, dict[RouteKey, SolverVar], list[Customer]]:

    # 1. Cắt giảm dữ liệu
    customers: list[Customer] = original_customers[:limit_nodes]

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
    x: dict[RouteKey, SolverVar] = {}
    t: dict[NodeVehicleKey, SolverVar] = {}
    u: dict[NodeVehicleKey, SolverVar] = {}

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
