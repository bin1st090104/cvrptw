from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple


__all__ = ("Problem",)

HEADER_PATTERN = re.compile(r"^\s*(.+?)\s+VEHICLE\s+NUMBER\s+CAPACITY\s+(\d+)\s+(\d+)", re.MULTILINE)
DATA_PATTERN = re.compile(r"^\s*\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Problem:
    name: str
    vehicle_capacities: List[int]
    time_matrix: List[List[float]]
    demands: List[int]
    time_windows: List[Tuple[int, int]]
    service_times: List[int]
    depot: Literal[0] = 0

    @classmethod
    def from_file(cls, path: Path) -> Problem:
        data = path.read_text(encoding="utf-8")
        match = HEADER_PATTERN.search(data)
        if match is None:
            raise RuntimeError(f"File {path} has an invalid header")

        name = match.group(1)

        try:
            vehicles_count = int(match.group(2))
            capacity = int(match.group(3))
        except ValueError as e:
            raise RuntimeError(f"File {path} has an invalid header") from e

        xs: List[int] = []
        ys: List[int] = []
        demands: List[int] = []
        time_windows: List[Tuple[int, int]] = []
        service_times: List[int] = []
        for match in DATA_PATTERN.finditer(data):
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                demand = int(match.group(3))
                ready_time = int(match.group(4))
                due_date = int(match.group(5))
                service_time = int(match.group(6))  # TODO: use this value
            except ValueError as e:
                raise RuntimeError(f"File {path} has invalid data") from e

            xs.append(x)
            ys.append(y)
            demands.append(demand)
            time_windows.append((ready_time, due_date))
            service_times.append(service_time)

        vehicle_capacities = [capacity] * vehicles_count
        time_matrix = [
            [
                ((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) ** 0.5
                for j in range(len(xs))
            ]
            for i in range(len(xs))
        ]
        return cls(
            name=name,
            vehicle_capacities=vehicle_capacities,
            time_matrix=time_matrix,
            demands=demands,
            time_windows=time_windows,
            service_times=service_times,
            depot=0,
        )
