"""
Insertion Heuristic for CVRPTW.

A constructive heuristic that iteratively inserts customers
into routes at the cheapest feasible position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from utils.solution import Solution
from generator import Instance


@dataclass
class InsertionCost:
    """Represents the cost of inserting a customer at a position."""
    customer: int
    route_idx: int
    position: int  # Insert after this index in route
    cost_increase: float
    new_route: List[int]


class InsertionSolver:
    """
    Cheapest Insertion Heuristic for CVRPTW.

    Builds routes by iteratively finding the customer and position
    that results in the minimum cost increase while respecting constraints.

    Variants:
    - 'cheapest': Insert at position with minimum cost increase
    - 'farthest': Prioritize farthest customers first (better for TSP-like)
    - 'regret': Use regret-based selection (difference between best and 2nd best)
    """

    def __init__(self, timeout: float = 1.0, variant: str = 'regret') -> None:
        self.timeout: float = timeout
        self.variant: str = variant  # 'cheapest', 'farthest', 'regret'

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

        # Initialize empty routes
        routes: List[List[int]] = [[start, end] for _ in range(K)]
        loads: List[int] = [0] * K
        unvisited = set(range(1, n + 1))

        while unvisited:
            if self.variant == 'regret':
                insertion = self._find_regret_insertion(
                    unvisited, routes, loads, dist, service, demand, a, b, Q, n, start, end
                )
            elif self.variant == 'farthest':
                insertion = self._find_farthest_insertion(
                    unvisited, routes, loads, dist, service, demand, a, b, Q, n, start, end
                )
            else:  # cheapest
                insertion = self._find_cheapest_insertion(
                    unvisited, routes, loads, dist, service, demand, a, b, Q, n, start, end
                )

            if insertion is None:
                # Try to open a new route (if possible by creating empty ones)
                return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

            # Perform insertion
            routes[insertion.route_idx] = insertion.new_route
            loads[insertion.route_idx] += demand[insertion.customer]
            unvisited.remove(insertion.customer)

        # Filter out empty routes
        final_routes = [r for r in routes if len(r) > 2]
        obj = self._compute_objective(dist, final_routes)
        return Solution(status="FEASIBLE", objective=obj, routes=final_routes)

    def _find_cheapest_insertion(
        self,
        unvisited: set,
        routes: List[List[int]],
        loads: List[int],
        dist: np.ndarray,
        service: np.ndarray,
        demand: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Q: int,
        n: int,
        start: int,
        end: int
    ) -> Optional[InsertionCost]:
        """Find the globally cheapest feasible insertion."""
        best: Optional[InsertionCost] = None

        for cust in unvisited:
            for ridx, route in enumerate(routes):
                if loads[ridx] + demand[cust] > Q:
                    continue

                insertion = self._best_position_in_route(
                    cust, route, ridx, dist, service, a, b, n, start, end
                )
                if insertion is not None:
                    if best is None or insertion.cost_increase < best.cost_increase:
                        best = insertion

        return best

    def _find_farthest_insertion(
        self,
        unvisited: set,
        routes: List[List[int]],
        loads: List[int],
        dist: np.ndarray,
        service: np.ndarray,
        demand: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Q: int,
        n: int,
        start: int,
        end: int
    ) -> Optional[InsertionCost]:
        """Select farthest customer from depot, then find cheapest position."""
        # Sort unvisited by distance from depot (farthest first)
        sorted_cust = sorted(unvisited, key=lambda c: -dist[start, c])

        for cust in sorted_cust:
            best_for_cust: Optional[InsertionCost] = None

            for ridx, route in enumerate(routes):
                if loads[ridx] + demand[cust] > Q:
                    continue

                insertion = self._best_position_in_route(
                    cust, route, ridx, dist, service, a, b, n, start, end
                )
                if insertion is not None:
                    if best_for_cust is None or insertion.cost_increase < best_for_cust.cost_increase:
                        best_for_cust = insertion

            if best_for_cust is not None:
                return best_for_cust

        return None

    def _find_regret_insertion(
        self,
        unvisited: set,
        routes: List[List[int]],
        loads: List[int],
        dist: np.ndarray,
        service: np.ndarray,
        demand: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Q: int,
        n: int,
        start: int,
        end: int
    ) -> Optional[InsertionCost]:
        """
        Use regret-2 heuristic: select customer with highest regret
        (difference between best and second-best insertion cost).
        """
        best_regret = -float('inf')
        best_insertion: Optional[InsertionCost] = None

        for cust in unvisited:
            insertions: List[InsertionCost] = []

            for ridx, route in enumerate(routes):
                if loads[ridx] + demand[cust] > Q:
                    continue

                insertion = self._best_position_in_route(
                    cust, route, ridx, dist, service, a, b, n, start, end
                )
                if insertion is not None:
                    insertions.append(insertion)

            if not insertions:
                continue

            # Sort by cost
            insertions.sort(key=lambda x: x.cost_increase)
            best_for_cust = insertions[0]

            # Compute regret (difference between 1st and 2nd best)
            if len(insertions) >= 2:
                regret = insertions[1].cost_increase - \
                    insertions[0].cost_increase
            else:
                # High regret for customers with only one option
                regret = float('inf')

            if regret > best_regret:
                best_regret = regret
                best_insertion = best_for_cust

        return best_insertion

    def _best_position_in_route(
        self,
        cust: int,
        route: List[int],
        ridx: int,
        dist: np.ndarray,
        service: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        n: int,
        start: int,
        end: int
    ) -> Optional[InsertionCost]:
        """Find the best position to insert customer in a route."""
        best: Optional[InsertionCost] = None

        # Try inserting after each position (except the last which is depot)
        for pos in range(len(route) - 1):
            prev_node = route[pos]
            next_node = route[pos + 1]

            # Cost increase
            cost_inc = (dist[prev_node, cust] + dist[cust, next_node]
                        - dist[prev_node, next_node])

            # Build new route
            new_route = route[:pos + 1] + [cust] + route[pos + 1:]

            # Check time window feasibility
            if not self._is_feasible(new_route, dist, service, a, b, n, start, end):
                continue

            if best is None or cost_inc < best.cost_increase:
                best = InsertionCost(
                    customer=cust,
                    route_idx=ridx,
                    position=pos,
                    cost_increase=cost_inc,
                    new_route=new_route
                )

        return best

    def _is_feasible(
        self,
        route: List[int],
        dist: np.ndarray,
        service: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        n: int,
        start: int,
        end: int
    ) -> bool:
        """Check if a route satisfies time window constraints."""
        current_time = float(a[start])

        for k in range(1, len(route)):
            prev, curr = route[k - 1], route[k]
            arrival = current_time + dist[prev, curr]

            # Wait if early
            service_start = max(arrival, a[curr])

            # Check deadline
            if service_start > b[curr]:
                return False

            current_time = service_start + service[curr]

        return True

    def _compute_objective(self, dist: np.ndarray, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
        return total
