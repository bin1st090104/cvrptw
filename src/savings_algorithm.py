"""
Clarke-Wright Savings Algorithm for CVRPTW.

A classic heuristic that computes savings from merging routes
and iteratively merges the most beneficial feasible pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
import numpy as np

from utils.solution import Solution
from generator import Instance


@dataclass
class RouteInfo:
    """Information about a route for quick access."""
    nodes: List[int]          # Full route [0, ..., n+1]
    load: int                 # Current load
    arrival_times: List[float]  # Arrival time at each node


class SavingsSolver:
    """
    Clarke-Wright Savings Algorithm for CVRPTW.

    Starts with direct routes (depot -> customer -> depot) and
    iteratively merges routes based on savings while respecting
    capacity and time window constraints.
    """

    def __init__(self, timeout: float = 1.0) -> None:
        self.timeout: float = timeout

    def solve(self, inst: Instance) -> Solution:
        n_nodes = int(inst.n_nodes)
        n = int(inst.n_customers)
        K = int(inst.K)
        Q = int(inst.Q)
        start, end = 0, n_nodes - 1

        dist = np.asarray(inst.dist, dtype=float)
        service = np.asarray(inst.service, dtype=float)
        demand = np.asarray(inst.demand, dtype=int)
        tw = np.asarray(inst.tw, dtype=float)
        a = tw[:, 0]
        b = tw[:, 1]

        # Step 1: Initialize with direct routes
        routes: Dict[int, RouteInfo] = {}  # route_id -> RouteInfo
        customer_route: Dict[int, int] = {}  # customer -> route_id

        for cust in range(1, n + 1):
            arrival_depot = float(a[start])
            arrival_cust = arrival_depot + dist[start, cust]
            service_start = max(arrival_cust, a[cust])

            # Check basic feasibility
            if arrival_cust > b[cust]:
                return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

            service_end = service_start + service[cust]
            arrival_end = service_end + dist[cust, end]

            if arrival_end > b[end]:
                return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

            routes[cust] = RouteInfo(
                nodes=[start, cust, end],
                load=int(demand[cust]),
                arrival_times=[arrival_depot, service_start, arrival_end]
            )
            customer_route[cust] = cust

        # Step 2: Compute savings
        # s(i,j) = d(0,i) + d(j,n+1) - d(i,j) for merging routes ending at i and starting at j
        savings: List[Tuple[float, int, int]] = []
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    s = dist[start, j] + dist[i, end] - dist[i, j]
                    if s > 0:
                        savings.append((s, i, j))

        # Sort by savings (descending)
        savings.sort(key=lambda x: -x[0])

        # Step 3: Merge routes greedily
        for saving, i, j in savings:
            ri = customer_route.get(i)
            rj = customer_route.get(j)

            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes.get(ri)
            route_j = routes.get(rj)

            if route_i is None or route_j is None:
                continue

            # i must be at end of route_i (before depot), j at start of route_j (after depot)
            if route_i.nodes[-2] != i or route_j.nodes[1] != j:
                continue

            # Check capacity
            new_load = route_i.load + route_j.load
            if new_load > Q:
                continue

            # Check vehicle limit (number of routes <= K)
            if len(routes) <= K:
                pass  # OK to merge

            # Merge routes: route_i + route_j (without duplicate depot)
            new_nodes = route_i.nodes[:-1] + route_j.nodes[1:]

            # Check time window feasibility
            new_arrivals = self._compute_arrivals(
                new_nodes, dist, service, a, b, start)
            if new_arrivals is None:
                continue

            # Merge is feasible - perform it
            new_route = RouteInfo(
                nodes=new_nodes,
                load=new_load,
                arrival_times=new_arrivals
            )

            # Update data structures
            del routes[rj]
            routes[ri] = new_route

            # Update customer_route for all customers in rj
            for node in route_j.nodes:
                if 1 <= node <= n:
                    customer_route[node] = ri

        # Step 4: Extract final routes
        final_routes = [info.nodes for info in routes.values()
                        if len(info.nodes) > 2]

        # Check vehicle limit
        if len(final_routes) > K:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        obj = self._compute_objective(dist, final_routes)
        return Solution(status="FEASIBLE", objective=obj, routes=final_routes)

    def _compute_arrivals(
        self,
        nodes: List[int],
        dist: np.ndarray,
        service: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        start: int
    ) -> Optional[List[float]]:
        """Compute arrival times for a route, returning None if infeasible."""
        arrivals = [float(a[start])]
        current_time = arrivals[0]

        for k in range(1, len(nodes)):
            prev, curr = nodes[k - 1], nodes[k]
            arrival = current_time + dist[prev, curr]

            # Wait if arriving early
            service_start = max(arrival, a[curr])

            # Check deadline
            if service_start > b[curr]:
                return None

            arrivals.append(service_start)
            current_time = service_start + service[curr]

        return arrivals

    def _compute_objective(self, dist: np.ndarray, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
        return total
