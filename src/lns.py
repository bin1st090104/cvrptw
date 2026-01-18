from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Set

import numpy as np
from ortools.linear_solver import pywraplp

from utils.solution import Solution
from generator import Instance

class LNSSolver:
    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout: float = timeout  # seconds

    def solve(self, instance: Instance) -> Solution:
        t0 = time.time()

        routes = self._init_greedy(instance)
        best_routes = routes
        best_obj = self._route_cost(instance, best_routes)

        rng = np.random.default_rng(12345)

        while time.time() - t0 < self.timeout:
            remaining = self.timeout - (time.time() - t0)
            if remaining <= 0.02:
                break

            R = self._destroy_set(instance, rng, frac=0.01)  # 1% nearest
            sol = self._repair_mip_cbc(instance, best_routes, R, time_limit=min(0.3, remaining))

            if sol.routes and sol.objective + 1e-9 < best_obj:
                best_obj = sol.objective
                best_routes = sol.routes

        if not best_routes:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        # LNS: never claim OPTIMAL
        return Solution(status="FEASIBLE", objective=float(best_obj), routes=best_routes)

    # ---------------- Init (no OR-Tools Routing) ----------------

    def _init_greedy(self, instance: Instance) -> List[List[int]]:
        """
        Simple greedy build:
        - Assign customers to vehicles sequentially (respect capacity roughly)
        - Order inside each route by nearest-neighbor
        This is not guaranteed feasible wrt TW, but gives a starting point for LNS.
        """
        n = int(instance.n_customers)
        K = int(instance.K)
        Q = int(instance.Q)

        dist = np.asarray(instance.dist, dtype=float)
        demand = np.asarray(instance.demand, dtype=int)

        depot_start = 0
        depot_end = int(instance.n_nodes) - 1

        customers = list(range(1, n + 1))
        # shuffle to diversify; deterministic seed can be added if you want
        np.random.default_rng(777).shuffle(customers)

        # rough assignment by capacity
        groups: List[List[int]] = [[] for _ in range(K)]
        loads = [0] * K
        for c in customers:
            # pick first vehicle that can take it, else least loaded
            placed = False
            for k in range(K):
                if loads[k] + demand[c] <= Q:
                    groups[k].append(c)
                    loads[k] += int(demand[c])
                    placed = True
                    break
            if not placed:
                k = int(np.argmin(loads))
                groups[k].append(c)
                loads[k] += int(demand[c])

        # order each group by NN from depot
        routes: List[List[int]] = []
        for g in groups:
            if not g:
                continue
            unvis = set(g)
            cur = depot_start
            seq = [depot_start]
            while unvis:
                nxt = min(unvis, key=lambda j: float(dist[cur, j]))
                unvis.remove(nxt)
                seq.append(int(nxt))
                cur = int(nxt)
            seq.append(depot_end)
            routes.append(seq)

        return routes or [[depot_start, depot_end]]

    # ---------------- Destroy ----------------

    def _destroy_set(self, instance: Instance, rng: np.random.Generator, frac: float) -> Set[int]:
        n = int(instance.n_customers)
        coords = np.asarray(instance.coords, dtype=float)

        seed = int(rng.integers(1, n + 1))
        p = max(1, int(np.ceil(frac * n)))

        cust = np.arange(1, n + 1, dtype=int)
        d = np.linalg.norm(coords[cust] - coords[seed], axis=1)
        order = cust[np.argsort(d)]
        R = set(int(x) for x in order[:p])
        R.add(seed)
        return R

    # ---------------- Repair (CBC only) ----------------

    def _repair_mip_cbc(
        self,
        instance: Instance,
        incumbent_routes: List[List[int]],
        R: Set[int],
        time_limit: float,
    ) -> Solution:
        n_nodes = int(instance.n_nodes)
        n = int(instance.n_customers)
        K = int(instance.K)
        Q = float(instance.Q)

        dist = np.asarray(instance.dist, dtype=float)
        service = np.asarray(instance.service, dtype=float)
        demand = np.asarray(instance.demand, dtype=float)
        tw = np.asarray(instance.tw, dtype=float)
        a, b = tw[:, 0], tw[:, 1]

        depot_start = 0
        depot_end = n_nodes - 1
        nodes = range(n_nodes)
        customers = range(1, n + 1)

        # IMPORTANT: CBC only (avoid SCIP segfault)
        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            return Solution(status="ERROR", objective=float("inf"), routes=[])

        solver.SetTimeLimit(max(1, int(time_limit * 1000)))

        A: List[Tuple[int, int]] = [
            (i, j)
            for i in nodes
            if i != depot_end
            for j in nodes
            if (i != j) and (j != depot_start) and not (i == depot_start and j == depot_end)
        ]

        H = float(np.max(b))
        M = H + float(np.max(service)) + float(np.max(dist)) + 1.0

        x: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        T: Dict[Tuple[int, int], pywraplp.Variable] = {}
        U: Dict[Tuple[int, int], pywraplp.Variable] = {}

        for k in range(K):
            for i in nodes:
                T[(k, i)] = solver.NumVar(float(a[i]), float(b[i]), f"T[{k},{i}]")
                U[(k, i)] = solver.NumVar(0.0, Q, f"U[{k},{i}]")
            for (i, j) in A:
                x[(k, i, j)] = solver.BoolVar(f"x[{k},{i},{j}]")

        solver.Minimize(
            solver.Sum(dist[i, j] * x[(k, i, j)] for k in range(K) for (i, j) in A)
        )

        # (1) each customer served exactly once (one outgoing overall)
        for i in customers:
            solver.Add(
                solver.Sum(x[(k, i, j)] for k in range(K) for j in nodes if (i, j) in A) == 1
            )

        # (2) flow conservation per vehicle at customers
        for k in range(K):
            for h in customers:
                solver.Add(
                    solver.Sum(x[(k, i, h)] for i in nodes if (i, h) in A)
                    ==
                    solver.Sum(x[(k, h, j)] for j in nodes if (h, j) in A)
                )

        # (3) start/end; if used then must go and return
        for k in range(K):
            out0 = solver.Sum(x[(k, depot_start, j)] for j in nodes if (depot_start, j) in A)
            inN = solver.Sum(x[(k, i, depot_end)] for i in nodes if (i, depot_end) in A)
            solver.Add(out0 <= 1)
            solver.Add(inN <= 1)
            solver.Add(out0 == inN)

        # (4) time precedence
        for k in range(K):
            for (i, j) in A:
                solver.Add(
                    T[(k, j)] >= T[(k, i)] + float(service[i]) + float(dist[i, j]) - M * (1 - x[(k, i, j)])
                )

        # (5) capacity propagation (simple)
        for k in range(K):
            solver.Add(U[(k, depot_start)] == 0.0)
            for (i, j) in A:
                if 1 <= j <= n:
                    solver.Add(
                        U[(k, j)] >= U[(k, i)] + float(demand[j]) - Q * (1 - x[(k, i, j)])
                    )

        # Fix incumbent arcs not touching destroyed set R
        for (k, i, j), val in self._fixed_arcs_from_incumbent(incumbent_routes, R).items():
            if (k, i, j) in x:
                x[(k, i, j)].SetBounds(val, val)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        routes = self._extract_routes(n_nodes, K, A, x)
        if not routes:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        return Solution(status="FEASIBLE", objective=float(solver.Objective().Value()), routes=routes)

    def _fixed_arcs_from_incumbent(self, routes: List[List[int]], R: Set[int]) -> Dict[Tuple[int, int, int], int]:
        fixed: Dict[Tuple[int, int, int], int] = {}
        for k, route in enumerate(routes):
            for i, j in zip(route[:-1], route[1:]):
                if (i in R) or (j in R):
                    continue
                fixed[(k, int(i), int(j))] = 1
        return fixed

    def _extract_routes(
        self,
        n_nodes: int,
        K: int,
        A: List[Tuple[int, int]],
        x: Dict[Tuple[int, int, int], pywraplp.Variable],
    ) -> List[List[int]]:
        depot_start = 0
        depot_end = n_nodes - 1
        nodes = range(n_nodes)

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
                cur = int(nxts[0])
                route.append(cur)
                if cur in seen:
                    break
                seen.add(cur)
            if route[0] == depot_start and route[-1] == depot_end:
                routes.append(route)
        return routes

    def _route_cost(self, instance: Instance, routes: List[List[int]]) -> float:
        dist = np.asarray(instance.dist, dtype=float)
        return float(sum(dist[int(i), int(j)] for r in routes for i, j in zip(r[:-1], r[1:])))