import math
from parse import Customer
from ortools.linear_solver import pywraplp

SolverVar = pywraplp.Variable
RouteKey = tuple[int, int, int]
NodeVehicleKey = tuple[int, int]

def euclidean(a: Customer, b: Customer) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def manhattan(a: Customer, b: Customer) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)

def build_distance(customers: list[Customer]) -> list[list[int]]:
    n = len(customers)
    dist: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = manhattan(customers[i], customers[j])
    return dist