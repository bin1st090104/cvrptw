from __future__ import annotations

from typing import List, TypedDict


__all__ = ("Solution",)


class Solution(TypedDict):
    cost: float
    feasible: bool
    routes: List[List[int]]
