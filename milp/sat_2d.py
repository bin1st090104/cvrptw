from .utils import *
from ortools.linear_solver import pywraplp
from parse import VehicleInfo, Customer


def solve_cvrptw_milp_sat_2d(
    original_customers: list[Customer],
    vehicle: VehicleInfo,
    limit_nodes: int = 25,
    time_limit_sec: int = 30
) -> tuple[pywraplp.Solver | None, int, dict[tuple[int, int], SolverVar], list[Customer]]:
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
