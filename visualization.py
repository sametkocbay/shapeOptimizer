#!/usr/bin/env python3
"""
visualization.py — Plotting utilities for the voxel-based CFD optimisation.

Provides:
  - ``plot_voxel_grid``           : colour-coded 2D voxel map
  - ``plot_optimization_history`` : downforce & geometry complexity over time
  - ``plot_stl_wireframe``        : lightweight 3D view of the exported STL
  - ``plot_comparison``           : side-by-side initial vs. best geometry

All functions accept an optional ``save_path`` to persist figures as PNG.

Author: VoxelCFD2D Project
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for WSL / headless)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import os


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
_FILLED_COLOUR = "#2196F3"   # material blue
_EMPTY_COLOUR  = "#ECEFF1"   # light blue-grey
_EDGE_COLOUR   = "#546E7A"   # dark grey
_ACCENT        = "#D32F2F"   # red for best-so-far
_PRIMARY       = "#1976D2"   # blue for current


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Voxel grid heat-map
# ──────────────────────────────────────────────────────────────────────────────
def plot_voxel_grid(grid, iteration=0, downforce=0.0,
                    save_path=None, show=False):
    """
    Render a 2D voxel grid as a coloured tile plot.

    Parameters
    ----------
    grid : np.ndarray   (nx, ny) with values 0 or 1
    iteration : int     current optimisation generation
    downforce : float   downforce result for this geometry
    save_path : str     path for saving the figure (PNG)
    show : bool         call plt.show() interactively
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    nx, ny = grid.shape
    for i in range(nx):
        for j in range(ny):
            colour = _FILLED_COLOUR if grid[i, j] == 1 else _EMPTY_COLOUR
            rect = mpatches.FancyBboxPatch(
                (i, j), 0.95, 0.95,
                boxstyle="round,pad=0.02",
                facecolor=colour,
                edgecolor=_EDGE_COLOUR,
                linewidth=0.5,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, nx + 0.5)
    ax.set_ylim(-0.5, ny + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (voxel index)", fontsize=12)
    ax.set_ylabel("Y (voxel index)", fontsize=12)
    ax.set_title(
        f"Voxel Grid — Iteration {iteration}\n"
        f"Downforce: {downforce:.4f} N  |  Filled: {int(np.sum(grid))}/{grid.size}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(range(0, nx + 1))
    ax.set_yticks(range(0, ny + 1))
    ax.grid(True, alpha=0.3, color="gray", linewidth=0.5)
    ax.tick_params(labelsize=8)

    filled_patch = mpatches.Patch(color=_FILLED_COLOUR, label="Filled (solid)")
    empty_patch  = mpatches.Patch(color=_EMPTY_COLOUR,  label="Empty (fluid)")
    ax.legend(handles=[filled_patch, empty_patch], loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [VIS] Grid plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Optimisation history
# ──────────────────────────────────────────────────────────────────────────────
def plot_optimization_history(history, save_path=None, show=False):
    """
    Two-panel figure: downforce over iterations  +  filled-cell count.

    Parameters
    ----------
    history : list of dict
        Each dict must have keys: iteration, downforce, best_downforce,
        filled_cells.
    """
    iters  = [h["iteration"] for h in history]
    df_cur = [h["downforce"] for h in history]
    df_bst = [h["best_downforce"] for h in history]
    filled = [h["filled_cells"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ── top panel: downforce ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(iters, df_cur, "o-", color=_PRIMARY, alpha=0.7,
             markersize=6, label="Current downforce", linewidth=1.5)
    ax1.plot(iters, df_bst, "s-", color=_ACCENT,
             markersize=6, label="Best downforce", linewidth=2)
    ax1.fill_between(iters, df_cur, alpha=0.08, color=_PRIMARY)
    ax1.set_ylabel("Downforce (N)", fontsize=12)
    ax1.set_title("Optimisation Progress — Downforce Maximisation",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # ── bottom panel: filled cells ───────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(iters, filled, color="#4CAF50", alpha=0.7, edgecolor="#2E7D32")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Filled cells", fontsize=12)
    ax2.set_title("Geometry Complexity", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [VIS] History plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  STL wireframe (lightweight 3D preview)
# ──────────────────────────────────────────────────────────────────────────────
def plot_stl_wireframe(stl_path, save_path=None, show=False):
    """
    Parse an ASCII STL and render it as a semi-transparent 3D surface.
    """
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

        verts_list = []
        with open(stl_path, "r") as fh:
            tri = []
            for line in fh:
                line = line.strip()
                if line.startswith("vertex"):
                    parts = line.split()
                    tri.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(tri) == 3:
                        verts_list.append(tri)
                        tri = []

        if not verts_list:
            print("  [VIS] No triangles found in STL — skipping wireframe plot.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        poly = Poly3DCollection(
            verts_list, alpha=0.25, facecolor=_FILLED_COLOUR,
            edgecolor="#1565C0", linewidth=0.2,
        )
        ax.add_collection3d(poly)

        all_v = np.array([v for tri in verts_list for v in tri])
        for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(all_v[:, idx].min(), all_v[:, idx].max())

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Voxel Shape — STL Preview", fontsize=13, fontweight="bold")

        if save_path:
            _ensure_dir(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [VIS] STL wireframe → {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    except Exception as exc:
        print(f"  [VIS] Could not render STL wireframe: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Side-by-side comparison (initial vs. best)
# ──────────────────────────────────────────────────────────────────────────────
def plot_comparison(grid_initial, grid_best, df_initial, df_best,
                    save_path=None, show=False):
    """
    Display the initial and best voxel grids side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, grid, label, df_val in [
        (axes[0], grid_initial, "Initial", df_initial),
        (axes[1], grid_best, "Best", df_best),
    ]:
        nx, ny = grid.shape
        for i in range(nx):
            for j in range(ny):
                colour = _FILLED_COLOUR if grid[i, j] == 1 else _EMPTY_COLOUR
                rect = mpatches.FancyBboxPatch(
                    (i, j), 0.95, 0.95,
                    boxstyle="round,pad=0.02",
                    facecolor=colour, edgecolor=_EDGE_COLOUR, linewidth=0.4,
                )
                ax.add_patch(rect)
        ax.set_xlim(-0.5, nx + 0.5)
        ax.set_ylim(-0.5, ny + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{label}  (Downforce {df_val:.4f} N)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Geometry Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [VIS] Comparison plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== visualization self-test ===")
    demo = np.zeros((10, 10), dtype=np.int32)
    demo[2:8, 3:7] = 1
    plot_voxel_grid(demo, iteration=0, downforce=1.23,
                    save_path="results/demo_grid.png")

    history = [
        {"iteration": i, "downforce": np.sin(i / 3) + 1,
         "best_downforce": 1 + i * 0.1, "filled_cells": 30 + i}
        for i in range(10)
    ]
    plot_optimization_history(history, save_path="results/demo_history.png")
    print("Self-test passed ✓")
