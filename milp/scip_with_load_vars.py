from .utils import RouteKey, SolverVar, build_distance, NodeVehicleKey
from ortools.linear_solver import pywraplp
from parse import VehicleInfo, Customer


def solve_cvrptw_milp_scip_with_load_vars(
    original_customers: list[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> tuple[pywraplp.Solver | None, int, dict[RouteKey, SolverVar], list[Customer]]:
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