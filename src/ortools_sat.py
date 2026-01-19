from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple, List
import numpy as np
from ortools.sat.python import cp_model

from utils.solution import Solution
from generator import Instance


class SATSolver:
    def __init__(self, timeout: float = 1.0) -> None:
        self.timeout: float = timeout  # seconds
        self.scale: int = 1_000_000

    def solve(self, inst: Instance) -> Solution:
        n_nodes = int(inst.n_nodes)
        n = int(inst.n_customers)
        K = int(inst.K)
        Q = int(inst.Q)
        start, end = 0, n_nodes - 1

        dist = np.asarray(inst.dist, float)
        service = np.asarray(inst.service, float)
        demand = np.asarray(inst.demand, int)
        tw = np.asarray(inst.tw, float)
        a = np.rint(tw[:, 0] * self.scale).astype(np.int64)
        b = np.rint(tw[:, 1] * self.scale).astype(np.int64)

        d = np.rint(dist * self.scale).astype(np.int64)
        s = np.rint(service * self.scale).astype(np.int64)

        nodes = range(n_nodes)
        customers = range(1, n + 1)

        # Available arcs A
        A: List[Tuple[int, int]] = [
            (i, j)
            for i in nodes
            if i != end
            for j in nodes
            if (i != j) and (j != start) and not (i == start and j == end)
        ]

        m = cp_model.CpModel()

        # Vars
        x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
        T: Dict[Tuple[int, int], cp_model.IntVar] = {}
        U: Dict[Tuple[int, int], cp_model.IntVar] = {}

        for k in range(K):
            for i in nodes:
                T[(k, i)] = m.NewIntVar(int(a[i]), int(b[i]), f"T[{k},{i}]")
                U[(k, i)] = m.NewIntVar(0, Q, f"U[{k},{i}]")
            for (i, j) in A:
                x[(k, i, j)] = m.NewBoolVar(f"x[{k},{i},{j}]")

        # Objective: min sum d_ij x_ij^k
        m.Minimize(sum(int(d[i, j]) * x[(k, i, j)] for k in range(K) for (i, j) in A))

        # (1) Each customer served exactly once: one outgoing arc overall
        for i in customers:
            m.Add(sum(x[(k, i, j)] for k in range(K) for j in nodes if (i, j) in A) == 1)

        # (2) Flow conservation per vehicle at customers
        for k in range(K):
            for h in customers:
                m.Add(
                    sum(x[(k, i, h)] for i in nodes if (i, h) in A)
                    ==
                    sum(x[(k, h, j)] for j in nodes if (h, j) in A)
                )

        # (3) Start/end; if used then must go and return
        for k in range(K):
            out0 = sum(x[(k, start, j)] for j in nodes if (start, j) in A)
            inN = sum(x[(k, i, end)] for i in nodes if (i, end) in A)
            m.Add(out0 <= 1)
            m.Add(inN <= 1)
            m.Add(out0 == inN)

        # (4) Time precedence: if x[k,i,j]=1 then Tj >= Ti + s_i + d_ij
        for k in range(K):
            for (i, j) in A:
                m.Add(T[(k, j)] >= T[(k, i)] + int(s[i]) + int(d[i, j])).OnlyEnforceIf(x[(k, i, j)])

        # (5) Capacity (simple, no activation): U0=0, and if x then Uj >= Ui + qj
        for k in range(K):
            m.Add(U[(k, start)] == 0)
            for (i, j) in A:
                if 1 <= j <= n:
                    m.Add(U[(k, j)] >= U[(k, i)] + int(demand[j])).OnlyEnforceIf(x[(k, i, j)])

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.timeout)
        solver.parameters.num_search_workers = 8  # can reduce if needed
        res = solver.Solve(m)

        if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            st = "INFEASIBLE" if res == cp_model.INFEASIBLE else "NO_SOLUTION"
            return Solution(status=st, objective=float("inf"), routes=[])

        # Extract routes
        routes: List[List[int]] = []
        for k in range(K):
            # find chosen arc out of start
            nxts = [j for j in nodes if (start, j) in A and solver.Value(x[(k, start, j)]) == 1]
            if not nxts:
                continue
            cur = start
            route = [cur]
            seen = {cur}

            while cur != end:
                cand = [j for j in nodes if (cur, j) in A and solver.Value(x[(k, cur, j)]) == 1]
                if not cand:
                    break
                cur = cand[0]
                route.append(cur)
                if cur in seen:
                    break
                seen.add(cur)

            if len(route) > 2 and route[0] == start and route[-1] == end:
                routes.append(route)

        obj = float(solver.ObjectiveValue()) / float(self.scale)
        return Solution(
            status="OPTIMAL" if res == cp_model.OPTIMAL else "FEASIBLE",
            objective=obj,
            routes=routes,
        )
