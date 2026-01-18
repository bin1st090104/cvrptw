from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List
import math

import numpy as np
from numpy.typing import NDArray


Mode = Literal["e", "m", "h"]  # easy / medium / hard


@dataclass(frozen=True, slots=True)
class Instance:
    # Sizes / sets
    n_customers: int           # n
    n_nodes: int               # n + 2 (0 .. n+1)
    K: int                     # number of vehicles
    Q: int                     # vehicle capacity

    # Data
    coords: NDArray[np.float64]     # (n+2, 2) in [0,1]
    dist: NDArray[np.float64]       # (n+2, n+2), also time
    demand: NDArray[np.int32]       # (n+2,), q_0=q_{n+1}=0
    service: NDArray[np.float64]    # (n+2,), s_0=s_{n+1}=0
    tw: NDArray[np.float64]         # (n+2, 2), [a_i,b_i]

    # (optional) one feasible solution used to generate tw
    routes: List[List[int]]         # each route is [0, ..., n+1]


def generate_cvrptw_dataset(
    seed: int,
    n_customers: int,          # interpreted as number of customers n
    n_instances: int,
    mode: Mode,
) -> list[Instance]:
    rng = np.random.default_rng(seed)
    n = int(n_customers)
    assert n >= 1
    assert n_instances >= 1
    assert mode in ("e", "m", "h")

    # Mode knobs (kept simple)
    veh_factor = {"e": 8, "m": 10, "h": 12}[mode]          # bigger => fewer vehicles
    tw_factor  = {"e": 0.60, "m": 0.30, "h": 0.15}[mode]   # time-window width as fraction of horizon
    s_max      = {"e": 0.02, "m": 0.03, "h": 0.05}[mode]   # service time magnitude

    out: list[Instance] = []

    for _ in range(n_instances):
        # ---- Nodes / coords ----
        # depot start at 0, depot return at n+1 (same coordinate for simplicity)
        depot = rng.random(2, dtype=np.float64)
        cust = rng.random((n, 2), dtype=np.float64)
        coords = np.vstack([depot[None, :], cust, depot[None, :]]).astype(np.float64)
        n_nodes = n + 2

        # ---- Dist (and time) ----
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float64)

        # ---- Choose K and build a baseline feasible routing structure ----
        K = max(1, int(math.ceil(n / veh_factor)))
        customers = np.arange(1, n + 1, dtype=np.int32)
        rng.shuffle(customers)

        groups: list[list[int]] = [list(g) for g in np.array_split(customers, K)]

        # ---- Demands: sample per group so each route can fit capacity ----
        # Keep it simple: choose Q large enough (mode-dependent tightness),
        # then rejection-sample demands for each group until sum <= Q.
        dem_low, dem_high = {"e": (1, 10), "m": (1, 15), "h": (1, 20)}[mode]
        avg_group = max(1, int(math.ceil(n / K)))
        tight = {"e": 1.60, "m": 1.25, "h": 1.05}[mode]
        Q = max(10, int(avg_group * (dem_low + dem_high) / 2 * tight))

        demand = np.zeros(n_nodes, dtype=np.int32)

        def sample_group_demands(m: int) -> NDArray[np.int32]:
            while True:
                d = rng.integers(dem_low, dem_high + 1, size=m, dtype=np.int32)
                if int(d.sum()) <= Q:
                    return d

        for g in groups:
            if not g:
                continue
            dvals = sample_group_demands(len(g))
            for node, dval in zip(g, dvals):
                demand[int(node)] = int(dval)

        # ---- Service times ----
        service = np.zeros(n_nodes, dtype=np.float64)
        service[1 : n + 1] = rng.random(n, dtype=np.float64) * float(s_max)

        # ---- Order within each group (nearest neighbor) and compute baseline arrivals ----
        routes: list[list[int]] = []
        arrival = np.zeros(n_nodes, dtype=np.float64)

        for g in groups:
            if not g:
                routes.append([0, n + 1])
                continue

            unvisited = set(int(x) for x in g)
            cur = 0
            seq: list[int] = [0]
            t = 0.0

            while unvisited:
                nxt = min(unvisited, key=lambda j: float(dist[cur, j]))
                unvisited.remove(nxt)
                t += float(dist[cur, nxt])  # travel
                arrival[nxt] = t
                t += float(service[nxt])    # service
                seq.append(nxt)
                cur = nxt

            t += float(dist[cur, n + 1])
            arrival[n + 1] = max(arrival[n + 1], t)
            seq.append(n + 1)
            routes.append(seq)

        # ---- Time windows derived from baseline (guarantees feasibility of these routes) ----
        H = float(arrival[n + 1] + 0.2)          # horizon buffer
        W = max(0.05, float(tw_factor) * H)      # window width
        halfW = 0.5 * W

        tw = np.zeros((n_nodes, 2), dtype=np.float64)
        tw[0] = (0.0, H)
        tw[n + 1] = (0.0, H)

        for i in range(1, n + 1):
            ai = max(0.0, float(arrival[i] - halfW))
            bi = float(arrival[i] + halfW)
            # ensure b_i >= a_i + s_i
            if bi < ai + float(service[i]):
                bi = ai + float(service[i])
            tw[i] = (ai, bi)

        out.append(
            Instance(
                n_customers=n,
                n_nodes=n_nodes,
                K=K,
                Q=Q,
                coords=coords,
                dist=dist,          # also time matrix
                demand=demand,
                service=service,
                tw=tw,
                routes=routes,
            )
        )

    return out
