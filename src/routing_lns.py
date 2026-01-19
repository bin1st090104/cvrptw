from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Sequence
import time
import math
import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from utils.solution import Solution
from generator import Instance


class LNSSolver:
    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout
        self.seed: int = 0
        self.destroy_frac: float = 0.25    # fraction of customers to remove each iteration
        self.sub_timeout: float = 50       # seconds per subproblem
        self.init_timeout: float = 10      # seconds for initial solution

    def solve(self, inst: Instance) -> Solution:
        rng = np.random.default_rng(self.seed)
        t_end = time.time() + float(self.timeout)

        # 1) Initial solution by Routing (greedy + tiny LS)
        cur_routes = self._routing_solve_full(inst, timeout=min(self.init_timeout, self.timeout))
        if cur_routes is None:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        cur_obj = self._objective(inst, cur_routes)
        best_routes, best_obj = cur_routes, cur_obj

        # 2) LNS loop
        while time.time() < t_end:
            n = int(inst.n_customers)
            if n <= 1:
                break

            r = max(1, int(round(self.destroy_frac * n)))
            remove = set(rng.choice(np.arange(1, n + 1), size=r, replace=False).tolist())

            # affected vehicles: those containing any removed customer
            veh_of = self._customer_to_vehicle(cur_routes)
            affected = sorted({veh_of[c] for c in remove if c in veh_of})
            if not affected:
                continue

            # customers involved in subproblem: all customers on affected vehicles
            sub_customers: List[int] = []
            for k in affected:
                for node in cur_routes[k]:
                    if 1 <= node <= n:
                        sub_customers.append(node)
            sub_customers = sorted(set(sub_customers))

            # solve subproblem on these customers with |affected| vehicles
            remaining_time = max(0.01, t_end - time.time())
            sub_t = min(self.sub_timeout, remaining_time)
            sub_routes = self._routing_solve_subset(inst, sub_customers, len(affected), timeout=sub_t)
            if sub_routes is None:
                continue

            # build candidate solution: replace affected routes, keep others
            cand_routes = [rt[:] for rt in cur_routes]
            # normalize: sub_routes is a list of used routes, may be fewer than len(affected)
            sub_routes_full = sub_routes + [[0, inst.n_nodes - 1]] * (len(affected) - len(sub_routes))
            for idx, k in enumerate(affected):
                cand_routes[k] = sub_routes_full[idx]

            cand_obj = self._objective(inst, cand_routes)
            if cand_obj < cur_obj - 1e-9:
                cur_routes, cur_obj = cand_routes, cand_obj
                if cand_obj < best_obj - 1e-9:
                    best_routes, best_obj = cand_routes, cand_obj

        return Solution(status="FEASIBLE", objective=float(best_obj), routes=self._used_routes(best_routes))

    # ---------------------- helpers ----------------------

    def _used_routes(self, routes_by_vehicle: List[List[int]]) -> List[List[int]]:
        return [r for r in routes_by_vehicle if len(r) > 2]

    def _objective(self, inst: Instance, routes_by_vehicle: List[List[int]]) -> float:
        dist = np.asarray(inst.dist, dtype=float)
        total = 0.0
        for r in routes_by_vehicle:
            for i in range(len(r) - 1):
                total += float(dist[r[i], r[i + 1]])
        return total

    def _customer_to_vehicle(self, routes_by_vehicle: List[List[int]]) -> Dict[int, int]:
        m: Dict[int, int] = {}
        for k, r in enumerate(routes_by_vehicle):
            for node in r:
                if node not in (0, r[-1]) and node != r[-1]:
                    m[node] = k
        return m

    # ---- Routing (full instance) ----
    def _routing_solve_full(self, inst: Instance, timeout: float) -> Optional[List[List[int]]]:
        n_nodes = int(inst.n_nodes)
        K = int(inst.K)
        start, end = 0, n_nodes - 1
        nodes = list(range(n_nodes))
        return self._routing_solve(inst, nodes, K, timeout)

    # ---- Routing (subset subproblem) ----
    def _routing_solve_subset(
        self, inst: "Instance", customers: Sequence[int], K_sub: int, timeout: float
    ) -> Optional[List[List[int]]]:
        n_nodes = int(inst.n_nodes)
        start, end = 0, n_nodes - 1
        # node list: [0] + customers + [n+1]
        nodes = [start] + list(customers) + [end]
        return self._routing_solve(inst, nodes, K_sub, timeout)

    def _routing_solve(
        self, inst: "Instance", nodes: List[int], K: int, timeout: float
    ) -> Optional[List[List[int]]]:
        # Build a compact sub-instance with reindexed nodes
        orig_start, orig_end = nodes[0], nodes[-1]
        assert orig_start == 0 and orig_end == int(inst.n_nodes) - 1

        idx_of: Dict[int, int] = {node: i for i, node in enumerate(nodes)}
        inv: List[int] = nodes[:]  # sub_index -> orig_node

        dist_full = np.asarray(inst.dist, dtype=float)
        serv_full = np.asarray(inst.service, dtype=float)
        dem_full = np.asarray(inst.demand, dtype=int)
        tw_full = np.asarray(inst.tw, dtype=float)
        a_full, b_full = tw_full[:, 0], tw_full[:, 1]

        m = len(nodes)
        start, end = 0, m - 1
        scale = 1000

        mgr = pywrapcp.RoutingIndexManager(m, K, [start] * K, [end] * K)
        routing = pywrapcp.RoutingModel(mgr)

        def cost_cb(i: int, j: int) -> int:
            ni, nj = inv[mgr.IndexToNode(i)], inv[mgr.IndexToNode(j)]
            return int(round(dist_full[ni, nj] * scale))

        cost_idx = routing.RegisterTransitCallback(cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

        def dem_cb(i: int) -> int:
            ni = inv[mgr.IndexToNode(i)]
            return int(dem_full[ni])

        dem_idx = routing.RegisterUnaryTransitCallback(dem_cb)
        routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [int(inst.Q)] * K, True, "Cap")

        def time_cb(i: int, j: int) -> int:
            ni, nj = inv[mgr.IndexToNode(i)], inv[mgr.IndexToNode(j)]
            return int(round((dist_full[ni, nj] + serv_full[ni]) * scale))

        time_idx = routing.RegisterTransitCallback(time_cb)
        horizon = int(math.ceil(float(np.max(b_full[inv])) * scale)) + 1
        routing.AddDimension(time_idx, horizon, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # Time windows: customers via NodeToIndex; depots via Start/End per vehicle (avoid segfault)
        for sub_node in range(1, m - 1):  # all customers in this subproblem
            orig = inv[sub_node]
            idx = mgr.NodeToIndex(sub_node)
            time_dim.CumulVar(idx).SetRange(int(round(a_full[orig] * scale)), int(round(b_full[orig] * scale)))

        orig0, origN = inv[start], inv[end]
        for k in range(K):
            s_idx = routing.Start(k)
            e_idx = routing.End(k)
            time_dim.CumulVar(s_idx).SetRange(int(round(a_full[orig0] * scale)), int(round(b_full[orig0] * scale)))
            time_dim.CumulVar(e_idx).SetRange(int(round(a_full[origN] * scale)), int(round(b_full[origN] * scale)))

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.time_limit.FromSeconds(max(1, int(math.ceil(timeout))))
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

        sol = routing.SolveWithParameters(params)
        if sol is None:
            return None

        routes: List[List[int]] = []
        for k in range(K):
            idx = routing.Start(k)
            r: List[int] = []
            while not routing.IsEnd(idx):
                r.append(inv[mgr.IndexToNode(idx)])
                idx = sol.Value(routing.NextVar(idx))
            r.append(inv[mgr.IndexToNode(idx)])

            if len(r) > 2:
                routes.append(r)

        return routes
