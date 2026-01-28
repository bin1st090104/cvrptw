"""
Genetic Algorithm for CVRPTW.

An evolutionary metaheuristic that maintains a population of solutions
and evolves them through selection, crossover, and mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import time
import numpy as np
from copy import deepcopy

from utils.solution import Solution
from generator import Instance


@dataclass
class Individual:
    """Represents an individual (solution) in the population."""
    routes: List[List[int]]
    fitness: float  # Lower is better (total distance)


class GeneticAlgorithmSolver:
    """
    Genetic Algorithm for CVRPTW.

    Features:
    - Giant tour representation with route-first, cluster-second
    - Order crossover (OX) for permutation-based GA
    - Mutation: swap, insert, 2-opt
    - Tournament selection
    - Elitism: Keep best individuals
    """

    def __init__(
        self,
        timeout: float = 10.0,
        population_size: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        tournament_size: int = 5,
        seed: int = 42
    ) -> None:
        self.timeout: float = timeout
        self.population_size: int = population_size
        self.elite_size: int = elite_size
        self.mutation_rate: float = mutation_rate
        self.tournament_size: int = tournament_size
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

        # Initialize population
        population = self._initialize_population(rng)
        if not population:
            return Solution(status="NO_SOLUTION", objective=float("inf"), routes=[])

        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness)
        best = deepcopy(population[0])

        generation = 0
        while time.time() < t_end:
            # Selection + Crossover + Mutation
            new_population: List[Individual] = []

            # Elitism: keep best individuals
            for i in range(min(self.elite_size, len(population))):
                new_population.append(deepcopy(population[i]))

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, rng)
                parent2 = self._tournament_selection(population, rng)

                # Crossover
                child_routes = self._crossover(
                    parent1.routes, parent2.routes, rng)

                # Mutation
                if rng.random() < self.mutation_rate:
                    child_routes = self._mutate(child_routes, rng)

                # Repair if needed and evaluate
                child_routes = self._repair_solution(child_routes)
                if child_routes is not None:
                    fitness = self._compute_objective(child_routes)
                    new_population.append(Individual(
                        routes=child_routes, fitness=fitness))

            population = new_population
            population.sort(key=lambda ind: ind.fitness)

            if population[0].fitness < best.fitness:
                best = deepcopy(population[0])

            generation += 1

        final_routes = [r for r in best.routes if len(r) > 2]
        return Solution(status="FEASIBLE", objective=best.fitness, routes=final_routes)

    def _initialize_population(self, rng: np.random.Generator) -> List[Individual]:
        """Create initial population with diverse solutions."""
        population: List[Individual] = []

        # Add nearest neighbor solution
        nn_routes = self._nearest_neighbor_init()
        if nn_routes is not None:
            fitness = self._compute_objective(nn_routes)
            population.append(Individual(routes=nn_routes, fitness=fitness))

        # Add random variations
        for _ in range(self.population_size - 1):
            routes = self._random_init(rng)
            if routes is not None:
                fitness = self._compute_objective(routes)
                population.append(Individual(routes=routes, fitness=fitness))

        return population

    def _nearest_neighbor_init(self) -> Optional[List[List[int]]]:
        """Nearest neighbor construction."""
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

        return routes if not unvisited else None

    def _random_init(self, rng: np.random.Generator) -> Optional[List[List[int]]]:
        """Random construction with feasibility check."""
        customers = list(range(1, self.n + 1))
        rng.shuffle(customers)

        routes: List[List[int]] = []
        unvisited = set(customers)

        while unvisited:
            route = [self.start]
            current = self.start
            current_time = float(self.a[self.start])
            current_load = 0

            # Try customers in shuffled order
            to_remove = []
            for cust in list(unvisited):
                if current_load + self.demand[cust] > self.Q:
                    continue
                arrival = current_time + self.dist[current, cust]
                if arrival > self.b[cust]:
                    continue
                service_end = max(arrival, self.a[cust]) + self.service[cust]
                if service_end + self.dist[cust, self.end] > self.b[self.end]:
                    continue

                # Add customer
                route.append(cust)
                to_remove.append(cust)
                arrival = current_time + self.dist[current, cust]
                current_time = max(arrival, self.a[cust]) + self.service[cust]
                current_load += self.demand[cust]
                current = cust

            for c in to_remove:
                unvisited.remove(c)

            route.append(self.end)
            if len(route) > 2:
                routes.append(route)
            elif unvisited:
                # Couldn't fit any customer - try starting fresh route
                pass

            # Safety: avoid infinite loop
            if len(routes) > self.K + self.n:
                return None

        return routes if not unvisited else None

    def _tournament_selection(
        self, population: List[Individual], rng: np.random.Generator
    ) -> Individual:
        """Select individual via tournament selection."""
        tournament = rng.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False
        )
        best = min(tournament, key=lambda i: population[i].fitness)
        return population[best]

    def _crossover(
        self,
        parent1: List[List[int]],
        parent2: List[List[int]],
        rng: np.random.Generator
    ) -> List[List[int]]:
        """Order crossover on giant tour representation."""
        # Convert to giant tours (just customers)
        tour1 = [c for r in parent1 for c in r if 1 <= c <= self.n]
        tour2 = [c for r in parent2 for c in r if 1 <= c <= self.n]

        if len(tour1) == 0 or len(tour2) == 0:
            return parent1

        n = len(tour1)

        # Select crossover points
        p1, p2 = sorted(rng.choice(n, size=2, replace=False))

        # OX crossover
        child = [None] * n
        child[p1:p2+1] = tour1[p1:p2+1]

        # Fill remaining from parent2
        used = set(child[p1:p2+1])
        pos = (p2 + 1) % n
        for c in tour2:
            if c not in used:
                while child[pos] is not None:
                    pos = (pos + 1) % n
                child[pos] = c
                used.add(c)

        # Convert back to routes
        return self._split_to_routes(child)

    def _split_to_routes(self, tour: List[int]) -> List[List[int]]:
        """Split giant tour into feasible routes."""
        routes: List[List[int]] = []
        route = [self.start]
        current_time = float(self.a[self.start])
        current_load = 0
        current = self.start

        for cust in tour:
            if cust is None:
                continue

            # Check if can add to current route
            can_add = True
            if current_load + self.demand[cust] > self.Q:
                can_add = False
            else:
                arrival = current_time + self.dist[current, cust]
                if arrival > self.b[cust]:
                    can_add = False
                else:
                    service_end = max(
                        arrival, self.a[cust]) + self.service[cust]
                    if service_end + self.dist[cust, self.end] > self.b[self.end]:
                        can_add = False

            if can_add:
                route.append(cust)
                arrival = current_time + self.dist[current, cust]
                current_time = max(arrival, self.a[cust]) + self.service[cust]
                current_load += self.demand[cust]
                current = cust
            else:
                # Start new route
                route.append(self.end)
                if len(route) > 2:
                    routes.append(route)

                route = [self.start, cust]
                arrival = self.a[self.start] + self.dist[self.start, cust]
                current_time = max(arrival, self.a[cust]) + self.service[cust]
                current_load = self.demand[cust]
                current = cust

        route.append(self.end)
        if len(route) > 2:
            routes.append(route)

        return routes

    def _mutate(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> List[List[int]]:
        """Apply mutation (swap or 2-opt)."""
        mutation_type = rng.integers(0, 3)

        if mutation_type == 0:
            # Swap within route
            return self._swap_mutation(routes, rng)
        elif mutation_type == 1:
            # 2-opt within route
            return self._two_opt_mutation(routes, rng)
        else:
            # Relocate between routes
            return self._relocate_mutation(routes, rng)

    def _swap_mutation(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> List[List[int]]:
        """Swap two customers within a route."""
        new_routes = deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if len(r) > 3]
        if not non_empty:
            return new_routes

        ridx = non_empty[rng.integers(len(non_empty))]
        route = new_routes[ridx]

        # Select two positions
        customers = [i for i in range(1, len(route) - 1)]
        if len(customers) < 2:
            return new_routes

        p1, p2 = rng.choice(customers, size=2, replace=False)
        route[p1], route[p2] = route[p2], route[p1]

        if self._is_feasible(route):
            new_routes[ridx] = route

        return new_routes

    def _two_opt_mutation(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> List[List[int]]:
        """Apply 2-opt reversal within a route."""
        new_routes = deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if len(r) > 3]
        if not non_empty:
            return new_routes

        ridx = non_empty[rng.integers(len(non_empty))]
        route = new_routes[ridx]

        i = rng.integers(1, len(route) - 2)
        j = rng.integers(i + 1, len(route) - 1)

        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]

        if self._is_feasible(new_route):
            new_routes[ridx] = new_route

        return new_routes

    def _relocate_mutation(
        self, routes: List[List[int]], rng: np.random.Generator
    ) -> List[List[int]]:
        """Move a customer to another route."""
        new_routes = deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if len(r) > 2]
        if len(non_empty) < 2:
            return new_routes

        # Source
        src_idx = non_empty[rng.integers(len(non_empty))]
        src_route = new_routes[src_idx]
        if len(src_route) <= 2:
            return new_routes

        cust_pos = rng.integers(1, len(src_route) - 1)
        cust = src_route[cust_pos]

        # Destination
        dst_idx = non_empty[rng.integers(len(non_empty))]
        if dst_idx == src_idx:
            return new_routes

        dst_route = new_routes[dst_idx]

        # Check capacity
        dst_load = sum(self.demand[c] for c in dst_route if 1 <= c <= self.n)
        if dst_load + self.demand[cust] > self.Q:
            return new_routes

        # Try inserting at random position
        insert_pos = rng.integers(1, len(dst_route))
        new_dst = dst_route[:insert_pos] + [cust] + dst_route[insert_pos:]

        if self._is_feasible(new_dst):
            new_routes[src_idx] = src_route[:cust_pos] + \
                src_route[cust_pos + 1:]
            new_routes[dst_idx] = new_dst

        return new_routes

    def _repair_solution(
        self, routes: List[List[int]]
    ) -> Optional[List[List[int]]]:
        """Ensure all customers are visited exactly once."""
        visited = set()
        for route in routes:
            for c in route:
                if 1 <= c <= self.n:
                    visited.add(c)

        missing = set(range(1, self.n + 1)) - visited
        if not missing:
            return routes

        # Try to insert missing customers
        new_routes = deepcopy(routes)
        for cust in missing:
            inserted = False
            for ridx, route in enumerate(new_routes):
                load = sum(self.demand[c] for c in route if 1 <= c <= self.n)
                if load + self.demand[cust] > self.Q:
                    continue

                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    if self._is_feasible(new_route):
                        new_routes[ridx] = new_route
                        inserted = True
                        break
                if inserted:
                    break

            if not inserted:
                # Create new route
                new_route = [self.start, cust, self.end]
                if self._is_feasible(new_route):
                    new_routes.append(new_route)
                else:
                    return None

        return new_routes

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
