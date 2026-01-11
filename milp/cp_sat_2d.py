from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python.cp_model import IntVar
from .utils import *
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from parse import VehicleInfo, Customer


def solve_cvrptw_milp_cp_sat_2d(
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
    # solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SAT")
    # if not solver:
    #     return None, pywraplp.Solver.NOT_SOLVED, {}, customers
    # solver.SetTimeLimit(time_limit_sec * 1000)

    x: dict[tuple[int, int], IntVar] = {}
    t: dict[int, IntVar] = {}
    u: dict[int, IntVar] = {}
    for i in range(N):
        t[i] = model.new_int_var(customers[i].ready_time, customers[i].due_date, f't[{i}]')
        u[i] = model.new_int_var(customers[i].demand, Q if i != 0 else 0, f'u[{i}]')
        for j in range(N):
            if i == j or j == 0 or i == N - 1 or (i == 0 and j == N - 1):
                continue
            x[i, j] = model.new_bool_var(f'x[{i}, {j}]')

    model.minimize(
        sum(
            dist[i][j] * x[i, j]
            for i in range(N)
            for j in range(N)
            if (i, j) in x
        )
    )

    for i in range(1, N - 1):
        model.add(
            sum(
                x[j, i]
                for j in range(0, N - 1)
                if i != j
            ) == 1
        )

    for (i, j) in x:
        model.add(t[j] >= t[i] + customers[i].service_time + dist[i][j]).only_enforce_if(x[i, j])
        model.add(u[j] >= u[i] + customers[j].demand).only_enforce_if(x[i, j])

    model.add(
        sum(
            x[0, i]
            for i in range(1, N - 1)
        ) <= vehicle.number
    )
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

    for i in range(1, N - 1):
        model.add(
            sum(
                x[i, j]
                for j in range(1, N)
                if i != j
            )
            ==
            sum(
                x[j, i]
                for j in range(0, N - 1)
                if i != j
            )
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    status = solver.solve(model)
    return solver, status, x, customers
