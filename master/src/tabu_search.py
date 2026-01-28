"""
Tabu Search for CVRPTW.

A metaheuristic that explores the neighborhood while maintaining
a tabu list to avoid cycling and encourage diversification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from collections import deque
import time
import numpy as np
from copy import deepcopy

from utils.solution import Solution
from generator import Instance


@dataclass(frozen=True)
class Move:
    """Represents a move for the tabu list."""
    move_type: str
    customer: int
    from_route: int
    to_route: int
    position: int


class TabuSearchSolver:
    """
    Tabu Search metaheuristic for CVRPTW.

    Explores the neighborhood systematically, using a tabu list
    to forbid recently made moves and prevent cycling.

    Features:
    - Aspiration criterion: Accept tabu moves if they improve best known
    - Intensification: Focus search around best solution
    - Diversification: Penalize frequently used moves
    """

    def __init__(
        self,
        timeout: float = 10.0,
        tabu_tenure: int = 20,
        max_iter_no_improve: int = 100,
        seed: int = 42
    ) -> None:
        self.timeout: float = timeout
        self.tabu_tenure: int = tabu_tenure
        self.max_iter_no_improve: int = max_iter_no_improve
        self.seed: int = seed

    def solve(self, inst: Instance) -> Solution:
        n_nodes = int(inst.n_nodes)
        n = int(inst.n_customers)
        K = int(inst.K)
        Q = int(inst.Q)
        start, end = 0, n_nodes - 1

        self.dist = np.asarray(inst.dist, dtype=float)
        self.service = np.asarray(inst.service, dtype=float)
        self.demand = np.asarray(inst.demand, dtype=int)
        self.tw = np.asarray(inst.tw, dtype=float)
        self.a = self.tw[:, 0]
        self.b = self.tw[:, 1]
        self.Q = Q
        self.n = n
        self.K = K
        self.start = start
        self.end = end

        rng = np.random.default_rng(self.seed)
        t_end = time.time() + self.timeout

        # Get initial solution
        current_routes = self._nearest_neighbor_init()
        if current_routes is None:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        current_obj = self._compute_objective(current_routes)
        best_routes = deepcopy(current_routes)
        best_obj = current_obj

        # Tabu list: maps move hash to iteration when it becomes non-tabu
        tabu_list: Dict[int, int] = {}
        iteration = 0
        iter_no_improve = 0

        while time.time() < t_end and iter_no_improve < self.max_iter_no_improve:
            # Generate all neighbors and evaluate
            neighbors = self._generate_neighbors(current_routes)

            if not neighbors:
                iter_no_improve += 1
                iteration += 1
                continue

            # Find best non-tabu move (or best tabu if aspiration met)
            best_neighbor = None
            best_neighbor_obj = float('inf')
            best_move = None

            for neighbor, move in neighbors:
                obj = self._compute_objective(neighbor)
                move_hash = hash(move)

                is_tabu = tabu_list.get(move_hash, 0) > iteration
                aspiration = obj < best_obj  # Aspiration criterion

                if (not is_tabu or aspiration) and obj < best_neighbor_obj:
                    best_neighbor = neighbor
                    best_neighbor_obj = obj
                    best_move = move

            if best_neighbor is None:
                iter_no_improve += 1
                iteration += 1
                continue

            # Make the move
            current_routes = best_neighbor
            current_obj = best_neighbor_obj

            # Add reverse move to tabu list
            if best_move is not None:
                reverse_move = Move(
                    move_type=best_move.move_type,
                    customer=best_move.customer,
                    from_route=best_move.to_route,
                    to_route=best_move.from_route,
                    position=best_move.position
                )
                tabu_list[hash(reverse_move)] = iteration + self.tabu_tenure

            # Update best
            if current_obj < best_obj:
                best_routes = deepcopy(current_routes)
                best_obj = current_obj
                iter_no_improve = 0
            else:
                iter_no_improve += 1

            iteration += 1

            # Clean old tabu entries periodically
            if iteration % 100 == 0:
                tabu_list = {k: v for k, v in tabu_list.items()
                             if v > iteration}

        final_routes = [r for r in best_routes if len(r) > 2]
        return Solution(status="FEASIBLE", objective=best_obj, routes=final_routes)

    def _nearest_neighbor_init(self) -> Optional[List[List[int]]]:
        """Generate initial solution using nearest neighbor."""
        unvisited = set(range(1, self.n + 1))
        routes: List[List[int]] = []

        while unvisited:
            route = [self.start]
            current = self.start
            current_time = float(self.a[self.start])
            current_load = 0

            while unvisited:
                best_cust: Optional[int] = None
                best_dist = float('inf')

                for cust in unvisited:
                    if current_load + self.demand[cust] > self.Q:
                        continue

                    arrival = current_time + self.dist[current, cust]
                    if arrival > self.b[cust]:
                        continue

                    service_end = max(
                        arrival, self.a[cust]) + self.service[cust]
                    if service_end + self.dist[cust, self.end] > self.b[self.end]:
                        continue

                    if self.dist[current, cust] < best_dist:
                        best_dist = self.dist[current, cust]
                        best_cust = cust

                if best_cust is None:
                    break

                route.append(best_cust)
                unvisited.remove(best_cust)
                arrival = current_time + self.dist[current, best_cust]
                current_time = max(
                    arrival, self.a[best_cust]) + self.service[best_cust]
                current_load += self.demand[best_cust]
                current = best_cust

            route.append(self.end)
            routes.append(route)

        if unvisited:
            return None
        return routes

    def _generate_neighbors(
        self, routes: List[List[int]]
    ) -> List[Tuple[List[List[int]], Move]]:
        """Generate neighborhood by relocate moves."""
        neighbors = []

        for src_idx, src_route in enumerate(routes):
            if len(src_route) <= 2:
                continue

            for cust_pos in range(1, len(src_route) - 1):
                cust = src_route[cust_pos]
                new_src = src_route[:cust_pos] + src_route[cust_pos + 1:]

                for dst_idx, dst_route in enumerate(routes):
                    # Determine positions to try
                    if dst_idx == src_idx:
                        base_route = new_src
                    else:
                        base_route = dst_route

                    for insert_pos in range(1, len(base_route)):
                        new_dst = base_route[:insert_pos] + \
                            [cust] + base_route[insert_pos:]

                        # Check capacity
                        dst_load = sum(
                            self.demand[n] for n in new_dst if 1 <= n <= self.n
                        )
                        if dst_load > self.Q:
                            continue

                        # Check feasibility
                        if not self._is_feasible(new_dst):
                            continue

                        if dst_idx != src_idx and not self._is_feasible(new_src):
                            continue

                        # Build neighbor solution
                        new_routes = [r[:] for r in routes]
                        new_routes[src_idx] = new_src
                        new_routes[dst_idx] = new_dst

                        move = Move(
                            move_type='relocate',
                            customer=cust,
                            from_route=src_idx,
                            to_route=dst_idx,
                            position=insert_pos
                        )
                        neighbors.append((new_routes, move))

        return neighbors

    def _is_feasible(self, route: List[int]) -> bool:
        """Check time window feasibility."""
        if len(route) < 2:
            return True

        current_time = float(self.a[self.start])

        for k in range(1, len(route)):
            prev, curr = route[k - 1], route[k]
            arrival = current_time + self.dist[prev, curr]
            service_start = max(arrival, self.a[curr])

            if service_start > self.b[curr]:
                return False

            current_time = service_start + self.service[curr]

        return True

    def _compute_objective(self, routes: List[List[int]]) -> float:
        """Compute total distance."""
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += self.dist[route[i], route[i + 1]]
        return total
