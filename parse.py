import re
from dataclasses import dataclass


@dataclass
class VehicleInfo:
    number: int
    capacity: int

@dataclass
class Customer:
    id: int
    x: int
    y: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int


def read_solomon_vrptw(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    instance_name = lines[0]

    # ---- VEHICLE ----
    vehicle_idx = lines.index("VEHICLE")
    vehicle_data = re.split(r"\s+", lines[vehicle_idx + 2])
    vehicle = VehicleInfo(
        number=int(vehicle_data[0]),
        capacity=int(vehicle_data[1]),
    )

    # ---- CUSTOMER ----
    customer_idx = lines.index("CUSTOMER")
    customers: list[Customer] = []

    for line in lines[customer_idx + 2:]:
        parts = re.split(r"\s+", line)
        customers.append(
            Customer(
                id=int(parts[0]),
                x=int(parts[1]),
                y=int(parts[2]),
                demand=int(parts[3]),
                ready_time=int(parts[4]),
                due_date=int(parts[5]),
                service_time=int(parts[6]),
            )
        )

    return instance_name, vehicle, customers

if __name__ == "__main__":

    name, vehicle, customers = read_solomon_vrptw("testcases/C101.txt")

    print(name)
    print(vehicle)
    print(customers[0])  # depot
    print(len(customers))  # 101 customers (0–100)

