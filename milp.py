import math
from typing import List, Tuple, Dict, Optional
from ortools.linear_solver import pywraplp

# Import các class data từ file parse của bạn
from parse import read_solomon_vrptw, VehicleInfo, Customer

# --- Type Aliases (Định nghĩa kiểu tắt cho gọn) ---
# Biến solver trong OR-Tools có kiểu là pywraplp.Variable
SolverVar = pywraplp.Variable
# Key cho biến x là (i, j, k)
RouteKey = Tuple[int, int, int]
# Key cho biến t, u là (i, k)
NodeVehicleKey = Tuple[int, int]


def euclidean(a: Customer, b: Customer) -> float:
    """Tính khoảng cách Euclidean giữa 2 khách hàng."""
    return math.hypot(a.x - b.x, a.y - b.y)


def build_distance(customers: List[Customer]) -> List[List[float]]:
    """Xây dựng ma trận khoảng cách."""
    n = len(customers)
    dist: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = euclidean(customers[i], customers[j])
    return dist


def solve_cvrptw_milp(
    original_customers: List[Customer],
    vehicle: VehicleInfo
) -> Tuple[Optional[pywraplp.Solver], int, Dict[RouteKey, SolverVar], List[Customer]]:
    """
    Giải bài toán CVRPTW bằng MILP.
    Trả về: (Solver, Status code, Dictionary biến X, Danh sách Customer đã thêm Dummy)
    """

    # 1. Cắt giảm dữ liệu để test nhanh
    LIMIT_NODES = 30
    customers: List[Customer] = original_customers[:LIMIT_NODES]

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

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        # Trả về dummy data nếu không khởi tạo được solver
        return None, pywraplp.Solver.NOT_SOLVED, {}, customers

    BIG_M = 100000

    # ---- Variables (Khai báo kiểu rõ ràng) ----

    # x[i, j, k]: Biến nhị phân
    x: Dict[RouteKey, SolverVar] = {}

    # t[i, k]: Thời gian đến (liên tục)
    t: Dict[NodeVehicleKey, SolverVar] = {}

    # u[i, k]: Tải trọng (liên tục)
    u: Dict[NodeVehicleKey, SolverVar] = {}

    for k in range(K):
        for i in range(n):
            t[i, k] = solver.NumVar(
                float(customers[i].ready_time),
                float(customers[i].due_date),
                f"t[{i},{k}]"
            )
            u[i, k] = solver.NumVar(0.0, float(Q), f"u[{i},{k}]")

            for j in range(n):
                if i == j: continue
                if j == 0: continue          # Không vào Start
                if i == n - 1: continue      # Không ra End

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

    # 1. Visit Constraints (1 -> n-2)
    for i in range(1, n - 1):
        solver.Add(
            solver.Sum(x[i, j, k] for k in range(K) for j in range(n) if (i, j, k) in x) == 1
        )

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
                    solver.Add(
                        t[j, k] >= t[i, k] + customers[i].service_time + dist[i][j] - BIG_M * (1 - x[i, j, k])
                    )

    # 5. Capacity
    for k in range(K):
        solver.Add(u[0, k] == 0)
        for i in range(n):
            for j in range(n):
                if (i, j, k) in x:
                    solver.Add(
                        u[j, k] >= u[i, k] + customers[j].demand - Q * (1 - x[i, j, k])
                    )

    print(f"Solving for {n-2} customers...")
    status: int = solver.Solve()
    return solver, status, x, customers


def extract_routes(
    x: Dict[RouteKey, SolverVar],
    customers: List[Customer],
    vehicle: VehicleInfo
) -> List[List[int]]:
    """Trích xuất lộ trình từ các biến quyết định."""

    n = len(customers)
    routes: List[List[int]] = []

    for k in range(vehicle.number):
        route: List[int] = []
        curr = 0

        while True:
            route.append(curr)
            if curr == n - 1:
                break

            found_next = False
            for j in range(n):
                if (curr, j, k) in x and x[curr, j, k].solution_value() > 0.5:
                    curr = j
                    found_next = True
                    break

            if not found_next:
                break

        # Chỉ lấy lộ trình có di chuyển (lớn hơn 2 điểm: Start -> ... -> End)
        if len(route) > 1 and route[-1] == n - 1:
            # Chuyển đổi ID của Dummy End Depot (n-1) về 0 để hiển thị đẹp hơn
            clean_route = [node_idx if node_idx != n-1 else 0 for node_idx in route]
            routes.append(clean_route)

    return routes


def main() -> None:
    try:
        # Giả sử file parse.py và testcase nằm đúng chỗ
        name, vehicle, customers = read_solomon_vrptw("testcases/C101.txt")
    except FileNotFoundError:
        print("Error: File not found.")
        return

    solver, status, x, new_customers = solve_cvrptw_milp(customers, vehicle)

    if solver and status == pywraplp.Solver.OPTIMAL:
        print("Status: OPTIMAL")
        print(f"Objective Value: {solver.Objective().Value()}")

        routes = extract_routes(x, new_customers, vehicle)
        for i, r in enumerate(routes):
            print(f"Vehicle {i}: {r}")
    else:
        print(f"Status: {status} (Not Optimal or Error)")


if __name__ == "__main__":
    main()