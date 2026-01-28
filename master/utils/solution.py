from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, List
import numpy as np
from numpy.typing import NDArray


SolveStatus = Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "NO_SOLUTION", "ERROR"]


@dataclass(slots=True)
class Solution:
    status: SolveStatus
    objective: float
    routes: List[List[int]]                 # each route: [0, ..., n+1]
    start_time: Optional[NDArray[np.float64]] = None  # (K, n_nodes), may be None
    load: Optional[NDArray[np.float64]] = None        # (K, n_nodes), may be None

    def __bool__(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE") and len(self.routes) > 0
