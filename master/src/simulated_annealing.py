"""
Simulated Annealing for CVRPTW.

A metaheuristic that improves an initial solution through
neighborhood moves, accepting worse solutions with decreasing probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import time
import math
import numpy as np
from copy import deepcopy

from utils.solution import Solution
from generator import Instance


class SimulatedAnnealingSolver:
    """
    Simulated Annealing metaheuristic for CVRPTW.

    Starts from an initial solution (nearest neighbor or random)
    and iteratively applies moves, accepting improvements always
    and worse solutions with probability exp(-delta/T).

    Moves include:
    - Relocate: Move a customer to another position
    - Swap: Exchange two customers between/within routes
    - 2-opt: Reverse a segment within a route
    - Or-opt: Move a sequence of 2-3 customers
    """

    def __init__(
        self,
        timeout: float = 10.0,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.9995,
        min_temp: float = 0.01,
        seed: int = 42
    ) -> None:
        self.timeout: float = timeout
        self.initial_temp: float = initial_temp
        self.cooling_rate: float = cooling_rate
        self.min_temp: float = min_temp
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
        self.start = start
        self.end = end

        rng = np.random.default_rng(self.seed)
        t_end = time.time() + self.timeout

        # Get initial solution using nearest neighbor
        current_routes = self._nearest_neighbor_init()
        if current_routes is None:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        current_obj = self._compute_objective(current_routes)
        best_routes = deepcopy(current_routes)
        best_obj = current_obj

        temp = self.initial_temp
        iterations = 0
        accepted = 0

        while time.time() < t_end and temp > self.min_temp:
            # Generate neighbor
            move_type = rng.integers(0, 4)

            if move_type == 0:
                neighbor = self._relocate_move(current_routes, rng)
            elif move_type == 1:
                neighbor = self._swap_move(current_routes, rng)
            elif move_type == 2:
                neighbor = self._two_opt_move(current_routes, rng)
            else:
                neighbor = self._or_opt_move(current_routes, rng)

            if neighbor is None:
                temp *= self.cooling_rate
                iterations += 1
                continue

            neighbor_obj = self._compute_objective(neighbor)
            delta = neighbor_obj - current_obj

            # Accept or reject
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_routes = neighbor
                current_obj = neighbor_obj
                accepted += 1

                if current_obj < best_obj:
                    best_routes = deepcopy(current_routes)
                    best_obj = current_obj

            temp *= self.cooling_rate
            iterations += 1

        # Clean up empty routes
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

    def _relocate_move(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> Optional[List[List[int]]]:
        """Move a customer to a different position."""
        # Find non-empty routes (with at least one customer)
        non_empty = [(i, r) for i, r in enumerate(routes) if len(r) > 2]
        if not non_empty:
            return None

        # Select source route and customer
        src_idx, src_route = non_empty[rng.integers(len(non_empty))]
        if len(src_route) <= 2:
            return None

        # Select customer to move (not depot)
        cust_pos = rng.integers(1, len(src_route) - 1)
        cust = src_route[cust_pos]

        # Select destination route
        dst_idx = rng.integers(len(routes))
        dst_route = routes[dst_idx]

        # Try all positions in destination
        best_pos: Optional[int] = None
        best_cost = float('inf')

        new_src = src_route[:cust_pos] + src_route[cust_pos + 1:]

        for pos in range(1, len(dst_route)):
            if dst_idx == src_idx:
                # Moving within same route
                new_dst = new_src[:pos] + [cust] + new_src[pos:]
            else:
                new_dst = dst_route[:pos] + [cust] + dst_route[pos:]

            # Check capacity
            dst_load = sum(self.demand[node]
                           for node in new_dst if 1 <= node <= self.n)
            if dst_load > self.Q:
                continue

            # Check time windows
            if not self._is_feasible(new_dst):
                continue

            cost = self._route_cost(new_dst)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        if best_pos is None:
            return None

        # Build new solution
        new_routes = deepcopy(routes)
        new_routes[src_idx] = new_src

        if dst_idx == src_idx:
            new_routes[dst_idx] = new_src[:best_pos] + \
                [cust] + new_src[best_pos:]
        else:
            new_routes[dst_idx] = dst_route[:best_pos] + \
                [cust] + dst_route[best_pos:]

        return new_routes

    def _swap_move(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> Optional[List[List[int]]]:
        """Swap two customers between routes."""
        non_empty = [(i, r) for i, r in enumerate(routes) if len(r) > 2]
        if len(non_empty) < 1:
            return None

        # Select two routes (can be same)
        r1_idx, r1 = non_empty[rng.integers(len(non_empty))]
        r2_idx, r2 = non_empty[rng.integers(len(non_empty))]

        if len(r1) <= 2 or len(r2) <= 2:
            return None

        # Select customers
        pos1 = rng.integers(1, len(r1) - 1)
        pos2 = rng.integers(1, len(r2) - 1)

        if r1_idx == r2_idx and pos1 == pos2:
            return None

        cust1, cust2 = r1[pos1], r2[pos2]

        # Build new routes
        new_routes = deepcopy(routes)

        if r1_idx == r2_idx:
            new_route = r1[:]
            new_route[pos1], new_route[pos2] = cust2, cust1

            if not self._is_feasible(new_route):
                return None
            new_routes[r1_idx] = new_route
        else:
            new_r1 = r1[:]
            new_r2 = r2[:]
            new_r1[pos1] = cust2
            new_r2[pos2] = cust1

            # Check capacity
            load1 = sum(self.demand[n] for n in new_r1 if 1 <= n <= self.n)
            load2 = sum(self.demand[n] for n in new_r2 if 1 <= n <= self.n)
            if load1 > self.Q or load2 > self.Q:
                return None

            if not self._is_feasible(new_r1) or not self._is_feasible(new_r2):
                return None

            new_routes[r1_idx] = new_r1
            new_routes[r2_idx] = new_r2

        return new_routes

    def _two_opt_move(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> Optional[List[List[int]]]:
        """Reverse a segment within a route (2-opt)."""
        non_empty = [(i, r) for i, r in enumerate(routes) if len(r) > 3]
        if not non_empty:
            return None

        ridx, route = non_empty[rng.integers(len(non_empty))]
        if len(route) <= 3:
            return None

        # Select two positions (not including depots)
        i = rng.integers(1, len(route) - 2)
        j = rng.integers(i + 1, len(route) - 1)

        # Reverse segment [i, j]
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]

        if not self._is_feasible(new_route):
            return None

        new_routes = deepcopy(routes)
        new_routes[ridx] = new_route
        return new_routes

    def _or_opt_move(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> Optional[List[List[int]]]:
        """Move a sequence of 1-3 customers to another position."""
        non_empty = [(i, r) for i, r in enumerate(routes) if len(r) > 2]
        if not non_empty:
            return None

        src_idx, src_route = non_empty[rng.integers(len(non_empty))]
        n_custs = len(src_route) - 2
        if n_custs < 1:
            return None

        # Select sequence length (1-3)
        seq_len = min(rng.integers(1, 4), n_custs)

        # Select start position of sequence
        start_pos = rng.integers(1, len(src_route) - seq_len)
        seq = src_route[start_pos:start_pos + seq_len]

        # Remove sequence from source
        new_src = src_route[:start_pos] + src_route[start_pos + seq_len:]

        # Select destination (can be same route)
        dst_idx = rng.integers(len(routes))

        if dst_idx == src_idx:
            dst_route = new_src
        else:
            dst_route = routes[dst_idx]

        # Try inserting at random position
        if len(dst_route) < 2:
            return None

        insert_pos = rng.integers(1, len(dst_route))
        new_dst = dst_route[:insert_pos] + seq + dst_route[insert_pos:]

        # Check capacity
        dst_load = sum(self.demand[n] for n in new_dst if 1 <= n <= self.n)
        if dst_load > self.Q:
            return None

        # Check feasibility
        if not self._is_feasible(new_dst):
            return None

        if dst_idx != src_idx and not self._is_feasible(new_src):
            return None

        new_routes = deepcopy(routes)
        new_routes[src_idx] = new_src
        new_routes[dst_idx] = new_dst

        return new_routes

    def _is_feasible(self, route: List[int]) -> bool:
        """Check time window feasibility of a route."""
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

    def _route_cost(self, route: List[int]) -> float:
        """Compute total distance of a route."""
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.dist[route[i], route[i + 1]]
        return cost

    def _compute_objective(self, routes: List[List[int]]) -> float:
        """Compute total objective (total distance)."""
        return sum(self._route_cost(r) for r in routes)
