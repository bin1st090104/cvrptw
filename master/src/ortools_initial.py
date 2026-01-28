from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List
import math
import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from utils.solution import Solution
from generator import Instance

class InitialSolver:
    def __init__(self, timeout: float = 1.0) -> None:
        self.timeout: float = timeout  # seconds

    def solve(self, inst: Instance) -> Solution:
        n_nodes = int(inst.n_nodes)          # = n + 2, nodes 0..n+1
        n = int(inst.n_customers)            # customers 1..n
        K = int(inst.K)
        start, end = 0, n_nodes - 1

        dist = np.asarray(inst.dist, dtype=float)
        service = np.asarray(inst.service, dtype=float)
        demand = np.asarray(inst.demand, dtype=int)
        tw = np.asarray(inst.tw, dtype=float)
        a = tw[:, 0]
        b = tw[:, 1]
        Q = int(inst.Q)

        scale = 1000  # Routing callbacks must return int

        mgr = pywrapcp.RoutingIndexManager(n_nodes, K, [start] * K, [end] * K)
        routing = pywrapcp.RoutingModel(mgr)

        # Cost: distance
        def cost_cb(i: int, j: int) -> int:
            ni, nj = mgr.IndexToNode(i), mgr.IndexToNode(j)
            return int(round(dist[ni, nj] * scale))

        cost_idx = routing.RegisterTransitCallback(cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

        # Capacity dimension
        def dem_cb(i: int) -> int:
            return int(demand[mgr.IndexToNode(i)])

        dem_idx = routing.RegisterUnaryTransitCallback(dem_cb)
        routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [Q] * K, True, "Cap")

        # Time dimension: travel + service(at from-node)
        def time_cb(i: int, j: int) -> int:
            ni, nj = mgr.IndexToNode(i), mgr.IndexToNode(j)
            return int(round((dist[ni, nj] + service[ni]) * scale))

        time_idx = routing.RegisterTransitCallback(time_cb)

        # Horizon: use max b_i as safe bound, scaled
        horizon = int(math.ceil(float(np.max(b)) * scale)) + 1

        routing.AddDimension(
            time_idx,
            horizon,   # slack max
            horizon,   # capacity (max time)
            False,     # don't force start cumul to zero; we respect depot TW
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # IMPORTANT: set TW correctly
        # - Customers: NodeToIndex is fine
        for node in range(1, n + 1):
            idx = mgr.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(int(round(a[node] * scale)), int(round(b[node] * scale)))

        # - Depots: must set per-vehicle Start/End indices
        for k in range(K):
            s_idx = routing.Start(k)
            e_idx = routing.End(k)
            time_dim.CumulVar(s_idx).SetRange(int(round(a[start] * scale)), int(round(b[start] * scale)))
            time_dim.CumulVar(e_idx).SetRange(int(round(a[end] * scale)), int(round(b[end] * scale)))

        # Search params
        params = pywrapcp.DefaultRoutingSearchParameters()
        # Duration supports seconds+nanos; keep it simple
        secs = max(1, int(math.ceil(self.timeout)))
        params.time_limit.FromSeconds(secs)

        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.UNSET

        sol = routing.SolveWithParameters(params)
        if sol is None:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        # Extract routes
        routes: List[List[int]] = []
        for k in range(K):
            idx = routing.Start(k)
            route: List[int] = []
            while not routing.IsEnd(idx):
                route.append(mgr.IndexToNode(idx))
                idx = sol.Value(routing.NextVar(idx))
            route.append(mgr.IndexToNode(idx))  # end

            # keep only used vehicles (has at least one customer)
            if len(route) > 2:
                routes.append(route)

        obj = float(sol.ObjectiveValue()) / scale
        return Solution(status="FEASIBLE", objective=obj, routes=routes)
