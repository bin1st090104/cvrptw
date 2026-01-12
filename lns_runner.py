from __future__ import annotations

from pathlib import Path

from lns import Problem
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


problem = Path("/workspaces/cvrptw/testcases/C101.txt")
data = Problem.from_file(problem)


manager = pywrapcp.RoutingIndexManager(
    len(data.time_matrix),
    len(data.vehicle_capacities),
    data.depot,
)


# @functools.cache
def time_callback_with_service(src: int, dst: int) -> float:
    src_node = manager.IndexToNode(src)
    dst_node = manager.IndexToNode(dst)
    return data.time_matrix[src_node][dst_node] + data.service_times[dst_node]


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

for vehicle in range(len(data.vehicle_capacities)):
    index = routing.Start(vehicle)
    time_dimension.CumulVar(index).SetRange(*data.time_windows[data.depot])

for vehicle in range(len(data.vehicle_capacities)):
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(vehicle)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(vehicle)))

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

solution = routing.SolveWithParameters(search_parameters)
