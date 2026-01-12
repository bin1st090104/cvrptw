from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

from lns import Problem, Solution
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def to_solution(assignment: pywrapcp.Assignment, *, data: Problem, routing: pywrapcp.RoutingModel) -> Solution:
    routes: List[List[int]] = []
    if assignment is not None:
        for vehicle in range(data.vehicles_count):
            index = routing.Start(vehicle)
            route: List[int] = []

            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = assignment.Value(routing.NextVar(index))

            route.append(manager.IndexToNode(index))

            if any(node != data.depot for node in route):
                routes.append(route)

    return Solution(
        cost=float("nan") if assignment is None else assignment.ObjectiveValue() / 100,
        feasible=assignment is not None,
        routes=routes,
    )


problem = Path(sys.argv[1])
data = Problem.from_file(problem)


manager = pywrapcp.RoutingIndexManager(
    len(data.time_matrix),
    data.vehicles_count,
    data.depot,
)


# @functools.cache
def time_callback_with_service(src: int, dst: int) -> int:
    src_node = manager.IndexToNode(src)
    dst_node = manager.IndexToNode(dst)
    return data.service_times[src_node] + data.time_matrix[src_node][dst_node]


def demand_callback(src: int) -> int:
    src_node = manager.IndexToNode(src)
    return data.demands[src_node]


routing = pywrapcp.RoutingModel(manager)
time_callback_with_service_index = routing.RegisterTransitCallback(time_callback_with_service)
demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

routing.SetArcCostEvaluatorOfAllVehicles(time_callback_with_service_index)

routing.AddDimension(
    time_callback_with_service_index,
    0,
    data.time_windows[data.depot][1],
    False,
    "Time",
)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    data.vehicle_capacities,
    True,
    "Capacity",
)

time_dimension = routing.GetDimensionOrDie("Time")
for index, time_window in enumerate(data.time_windows):
    if index == data.depot:
        continue

    node_index = manager.NodeToIndex(index)
    time_dimension.CumulVar(node_index).SetRange(*time_window)

for vehicle in range(data.vehicles_count):
    index = routing.Start(vehicle)
    time_dimension.CumulVar(index).SetRange(*data.time_windows[data.depot])

for vehicle in range(data.vehicles_count):
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(vehicle)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(vehicle)))

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

assignment = routing.SolveWithParameters(search_parameters)
solution = to_solution(assignment, data=data, routing=routing)
print(solution)
