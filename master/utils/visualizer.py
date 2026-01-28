from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, save_dir: str = "images/", use: bool = True, verbose: bool = False) -> None:
        self.save_dir = save_dir
        self.use = use
        self.verbose = verbose
        self._counter = 0

    def plot_solution(self, instance, solution, filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize CVRPTW solution on 2D square [0,1]x[0,1].
        - If self.use is False: do nothing, return None.
        - If self.verbose is True: annotate each node with its [a,b] time window.
        - Each route gets a different color.
        - Save figure into self.save_dir and return saved path.
        """
        if not self.use:
            return None

        os.makedirs(self.save_dir, exist_ok=True)

        coords = np.asarray(instance.coords, dtype=float)  # (n+2, 2)
        tw = np.asarray(getattr(instance, "tw", None), dtype=float) if getattr(instance, "tw", None) is not None else None

        routes = getattr(solution, "routes", []) or []
        status = getattr(solution, "status", "UNKNOWN")
        obj = getattr(solution, "objective", float("nan"))

        fig, ax = plt.subplots(figsize=(7, 7))

        # Canvas settings: unit square
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Plot all nodes (customers)
        n_nodes = coords.shape[0]
        depot_start = 0
        depot_end = n_nodes - 1

        cust_idx = np.arange(1, n_nodes - 1)
        ax.scatter(coords[cust_idx, 0], coords[cust_idx, 1], s=30, marker="o", zorder=2)

        # Plot depot(s)
        ax.scatter(coords[depot_start, 0], coords[depot_start, 1], s=140, marker="s", zorder=3)
        # If return depot is same coord, don't double-plot; still label it.
        if not np.allclose(coords[depot_start], coords[depot_end]):
            ax.scatter(coords[depot_end, 0], coords[depot_end, 1], s=140, marker="s", zorder=3)

        # Optional TW annotations
        if self.verbose and tw is not None and tw.shape == (n_nodes, 2):
            for i in range(n_nodes):
                a_i, b_i = float(tw[i, 0]), float(tw[i, 1])
                x, y = coords[i]
                ax.text(
                    x + 0.008,
                    y - 0.020,
                    f"[{a_i:.2f},{b_i:.2f}]",
                    fontsize=7,
                    zorder=5,
                )

        # Plot routes (each route a different color)
        cmap = plt.get_cmap("tab20")
        for r_idx, route in enumerate(routes):
            if not route or len(route) < 2:
                continue
            color = cmap(r_idx % cmap.N)

            pts = coords[np.array(route, dtype=int)]
            ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0, color=color, zorder=4)

            # Arrowheads for direction (lightweight)
            for u, v in zip(route[:-1], route[1:]):
                x1, y1 = coords[int(u)]
                x2, y2 = coords[int(v)]
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
                    zorder=4,
                )

        ax.set_title(f"CVRPTW Solution | status={status} | obj={obj:.4f} | routes={len(routes)}")

        # Save
        if filename is None:
            self._counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cvrptw_{ts}_{self._counter:04d}.png"

        path = os.path.join(self.save_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

        return path
