from ortools.sat.python.cp_model_helper import CpSolverStatus
from ortools.sat.python.cp_model import IntVar
from .utils import *
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from parse import VehicleInfo, Customer


def solve_cvrptw_milp_cp_sat_2d_optimized(
    original_customers: list[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> tuple[cp_model.CpSolver | None, CpSolverStatus, dict[tuple[int, int], IntVar], list[Customer]]:
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

    model = cp_model.CpModel()

    x: dict[tuple[int, int], IntVar] = {}
    t: dict[int, IntVar] = {}
    u: dict[int, IntVar] = {}
    for i in range(N):
        # thời gian xe đến phục vụ khách hàng i
        t[i] = model.new_int_var(customers[i].ready_time, customers[i].due_date, f't[{i}]')
        # tải trọng xe sau khi phục vụ khách hàng i
        u[i] = model.new_int_var(customers[i].demand, Q if i != 0 else 0, f'u[{i}]')
        for j in range(N):
            if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                continue
            if customers[i].ready_time + customers[i].service_time + dist[i][j] > customers[j].due_date:
                continue
            if customers[i].ready_time + customers[i].service_time + dist[i][j] + customers[j].service_time + dist[j][N - 1] > customers[N - 1].due_date:
                continue
            # biến quyết định xe có đi từ i đến j hay không
            x[i, j] = model.new_bool_var(f'x[{i}, {j}]')

    # mục tiêu: minimize tổng quãng đường đi
    model.minimize(
        sum(
            dist[i][j] * x[i, j]
            for i in range(N)
            for j in range(N)
            if (i, j) in x
        )
    )

    # ràng buộc mỗi khách hàng chỉ được phục vụ bởi đúng 1 xe
    for i in range(1, N - 1):
        model.add(
            sum(
                x[j, i]
                for j in range(0, N - 1)
                if (j, i) in x
            ) == 1
        )

    for (i, j) in x:
        # ràng buộc thời gian
        model.add(t[j] >= t[i] + customers[i].service_time + dist[i][j]).only_enforce_if(x[i, j])
        # ràng buộc tải trọng
        model.add(u[j] >= u[i] + customers[j].demand).only_enforce_if(x[i, j])

    # ràng buộc số xe
    model.add(
        sum(
            x[0, i]
            for i in range(1, N - 1)
        ) <= vehicle.number
    )

    # ràng buộc cân bằng luồng số xe ra từ 0 = số xe vào N - 1
    model.add(
        sum(
            x[0, i]
            for i in range(1, N - 1)
        )
        ==
        sum(
            x[i, N - 1]
            for i in range(1, N - 1)
        )
    )

    # ràng buộc cân bằng luồng cho mỗi khách hàng
    for i in range(1, N - 1):
        model.add(
            sum(
                x[i, j]
                for j in range(1, N)
                if (i, j) in x
            )
            == 1
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    status = solver.solve(model)
    return solver, status, x, customers
