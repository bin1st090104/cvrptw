"""
Nearest Neighbor Heuristic for CVRPTW.

A simple constructive heuristic that builds routes by always selecting
the nearest feasible (capacity + time window) unvisited customer.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from utils.solution import Solution
from generator import Instance


class NearestNeighborSolver:
    """
    Nearest Neighbor heuristic for CVRPTW.

    Builds routes sequentially by always choosing the nearest unvisited
    customer that satisfies capacity and time window constraints.
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
        a = tw[:, 0]  # earliest arrival
        b = tw[:, 1]  # latest arrival

        unvisited = set(range(1, n + 1))  # customers 1..n
        routes: List[List[int]] = []

        for _ in range(K):
            if not unvisited:
                break

            route = [start]
            current = start
            current_time = float(a[start])
            current_load = 0

            while unvisited:
                # Find nearest feasible customer
                best_customer: Optional[int] = None
                best_dist = float('inf')

                for cust in unvisited:
                    # Check capacity
                    if current_load + demand[cust] > Q:
                        continue

                    # Check time window feasibility
                    arrival_time = current_time + dist[current, cust]
                    if arrival_time > b[cust]:
                        # Cannot arrive before deadline
                        continue

                    # Check if we can return to depot on time
                    service_end = max(arrival_time, a[cust]) + service[cust]
                    return_time = service_end + dist[cust, end]
                    if return_time > b[end]:
                        continue

                    # Feasible - check if nearest
                    if dist[current, cust] < best_dist:
                        best_dist = dist[current, cust]
                        best_customer = cust

                if best_customer is None:
                    break

                # Add customer to route
                route.append(best_customer)
                unvisited.remove(best_customer)

                arrival = current_time + dist[current, best_customer]
                current_time = max(
                    arrival, a[best_customer]) + service[best_customer]
                current_load += demand[best_customer]
                current = best_customer

            route.append(end)
            if len(route) > 2:
                routes.append(route)

        # Check if all customers were visited
        if unvisited:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        obj = self._compute_objective(dist, routes)
        return Solution(status="FEASIBLE", objective=obj, routes=routes)

    def _compute_objective(self, dist: np.ndarray, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
        return total
