from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple, List
import numpy as np
from ortools.linear_solver import pywraplp

from utils.solution import Solution
from generator import Instance


class ExactSolver:
    def __init__(self, timeout: float = 1.0) -> None:
        self.timeout: float = timeout  # seconds

    def solve(self, instance: Instance) -> Solution:
        n_nodes = int(instance.n_nodes)
        n = int(instance.n_customers)
        K = int(instance.K)
        Q = float(instance.Q)

        dist = np.asarray(instance.dist, dtype=float)      # also time
        service = np.asarray(instance.service, dtype=float)
        demand = np.asarray(instance.demand, dtype=float)
        tw = np.asarray(instance.tw, dtype=float)
        a, b = tw[:, 0], tw[:, 1]

        solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            return Solution(status="ERROR", objective=float("inf"), routes=[])

        solver.SetTimeLimit(int(self.timeout * 1000))

        depot_start = 0
        depot_end = n_nodes - 1
        nodes = range(n_nodes)
        customers = range(1, n + 1)

        # Available arcs A
        A: List[Tuple[int, int]] = [
            (i, j)
            for i in nodes
            if i != depot_end
            for j in nodes
            if (i != j) and (j != depot_start) and not (i == depot_start and j == depot_end)
        ]

        # Big-M for time precedence
        H = float(np.max(b))
        M = H + float(np.max(service)) + float(np.max(dist)) + 1.0

        # Variables
        x: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        T: Dict[Tuple[int, int], pywraplp.Variable] = {}
        U: Dict[Tuple[int, int], pywraplp.Variable] = {}

        for k in range(K):
            for i in nodes:
                T[(k, i)] = solver.NumVar(float(a[i]), float(b[i]), f"T[{k},{i}]")
                U[(k, i)] = solver.NumVar(0.0, Q, f"U[{k},{i}]")
            for (i, j) in A:
                x[(k, i, j)] = solver.BoolVar(f"x[{k},{i},{j}]")

        # Objective
        solver.Minimize(
            solver.Sum(dist[i, j] * x[(k, i, j)] for k in range(K) for (i, j) in A)
        )

        # (1) Each customer served exactly once (one outgoing arc overall)
        for i in customers:
            solver.Add(
                solver.Sum(x[(k, i, j)] for k in range(K) for j in nodes if (i, j) in A) == 1
            )

        # (2) Flow conservation per vehicle at each customer
        for k in range(K):
            for h in customers:
                solver.Add(
                    solver.Sum(x[(k, i, h)] for i in nodes if (i, h) in A)
                    ==
                    solver.Sum(x[(k, h, j)] for j in nodes if (h, j) in A)
                )

        # (3) Start / end at depots; if used then must go and return
        for k in range(K):
            out0 = solver.Sum(x[(k, depot_start, j)] for j in nodes if (depot_start, j) in A)
            inN = solver.Sum(x[(k, i, depot_end)] for i in nodes if (i, depot_end) in A)
            solver.Add(out0 <= 1)
            solver.Add(inN <= 1)
            solver.Add(out0 == inN)

        # (4) Time precedence along arcs
        for k in range(K):
            for (i, j) in A:
                solver.Add(
                    T[(k, j)] >= T[(k, i)] + float(service[i]) + float(dist[i, j]) - M * (1 - x[(k, i, j)])
                )

        # (5) Capacity / load propagation (no activation constraint)
        for k in range(K):
            solver.Add(U[(k, depot_start)] == 0.0)
            for (i, j) in A:
                if 1 <= j <= n:  # only when arriving at a customer
                    solver.Add(
                        U[(k, j)] >= U[(k, i)] + float(demand[j]) - Q * (1 - x[(k, i, j)])
                    )

        # Solve
        status = solver.Solve()
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.NOT_SOLVED: "NO_SOLUTION",
        }
        st = status_map.get(status, "ERROR")
        if st not in ("OPTIMAL", "FEASIBLE"):
            return Solution(status=st, objective=float("inf"), routes=[])

        # Extract routes from x
        xval = {(k, i, j): x[(k, i, j)].solution_value() for (k, i, j) in x.keys()}
        routes: List[List[int]] = []

        for k in range(K):
            start_next = [j for j in nodes if (depot_start, j) in A and xval.get((k, depot_start, j), 0.0) > 0.5]
            if not start_next:
                continue
            cur = depot_start
            route = [cur]
            seen = {cur}

            while cur != depot_end:
                nxts = [j for j in nodes if (cur, j) in A and xval.get((k, cur, j), 0.0) > 0.5]
                if not nxts:
                    break
                cur = nxts[0]
                route.append(cur)
                if cur in seen:
                    break
                seen.add(cur)

            if route[0] == depot_start and route[-1] == depot_end:
                routes.append(route)

        return Solution(
            status=st,
            objective=float(solver.Objective().Value()),
            routes=routes,
        )
