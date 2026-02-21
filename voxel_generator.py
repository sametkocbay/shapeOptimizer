#!/usr/bin/env python3
"""
voxel_generator.py — Generate 2D voxel grids and export as STL for OpenFOAM.

This module provides functions to:
  - Create random or shaped 2D binary voxel grids
  - Convert voxel grids to watertight STL surfaces
  - Export STL files compatible with OpenFOAM's snappyHexMesh

The voxel grid lives within a bounding box and is extruded slightly in Z
to create a thin 3D body suitable for 2D OpenFOAM simulations.

Grid convention:
  grid[i, j]  →  i = x-index, j = y-index
  A value of 1 means the cell is FILLED (solid), 0 means EMPTY (fluid).

Author: VoxelCFD2D Project
"""

import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────────
# Default geometric parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_SIZE = (20, 20)        # (nx, ny) number of voxels
DEFAULT_VOXEL_SIZE = 0.025          # metres per voxel edge
DEFAULT_ORIGIN = (0.0, 0.1)         # (x, y) world-coordinate origin of grid
DEFAULT_Z_THICKNESS = 0.01          # extrusion depth for pseudo-2D
DEFAULT_FILL_FRACTION = 0.3         # initial random fill ratio


# ──────────────────────────────────────────────────────────────────────────────
# Grid generators
# ──────────────────────────────────────────────────────────────────────────────
def generate_random_grid(grid_size=DEFAULT_GRID_SIZE,
                         fill_fraction=DEFAULT_FILL_FRACTION,
                         seed=None):
    """
    Generate a random 2D binary voxel grid.

    Parameters
    ----------
    grid_size : tuple of int
        (nx, ny) dimensions of the grid.
    fill_fraction : float
        Probability that any cell is filled (0–1).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Binary 2D array of shape (nx, ny).
    """
    rng = np.random.default_rng(seed)
    grid = (rng.random(grid_size) < fill_fraction).astype(np.int32)
    return grid


def generate_airfoil_like_grid(grid_size=DEFAULT_GRID_SIZE):
    """
    Generate a simple wing / airfoil-like cross-section on the voxel grid.

    The shape is loosely inspired by a NACA-style thickness distribution,
    offset downward so it sits in the lower portion of the grid (ground
    effect positioning).

    Parameters
    ----------
    grid_size : tuple of int
        (nx, ny) dimensions of the grid.

    Returns
    -------
    np.ndarray
        Binary 2D array with the wing shape filled.
    """
    nx, ny = grid_size
    grid = np.zeros((nx, ny), dtype=np.int32)

    for i in range(nx):
        x_frac = i / nx
        # Thickness distribution (NACA-like)
        thickness = int(
            ny * 0.3 * (1.0 - x_frac) * np.sqrt(max(x_frac, 0.01))
        )
        centre = ny // 3  # push shape toward the bottom (ground effect)
        y_start = max(0, centre - thickness // 2)
        y_end = min(ny, centre + thickness // 2 + 1)
        grid[i, y_start:y_end] = 1

    return grid


def generate_wedge_grid(grid_size=DEFAULT_GRID_SIZE):
    """
    Generate a simple wedge / ramp shape.

    Parameters
    ----------
    grid_size : tuple of int

    Returns
    -------
    np.ndarray
    """
    nx, ny = grid_size
    grid = np.zeros((nx, ny), dtype=np.int32)
    for i in range(nx):
        height = max(1, int(ny * 0.5 * (1.0 - i / nx)))
        grid[i, :height] = 1
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def ensure_connected(grid):
    """
    Guarantee at least one filled cell exists (prevents empty STL).

    If the grid contains no filled cells, fill the centre cell.

    Parameters
    ----------
    grid : np.ndarray  (modified in-place)

    Returns
    -------
    np.ndarray
    """
    if np.sum(grid) == 0:
        cx, cy = grid.shape[0] // 2, grid.shape[1] // 2
        grid[cx, cy] = 1
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# STL export
# ──────────────────────────────────────────────────────────────────────────────
def voxel_grid_to_stl(grid, filename,
                      voxel_size=DEFAULT_VOXEL_SIZE,
                      origin=DEFAULT_ORIGIN,
                      z_thickness=DEFAULT_Z_THICKNESS):
    """
    Convert a 2D binary voxel grid into a **watertight** ASCII STL file.

    The grid is extruded from z = 0 to z = z_thickness.  Only external
    faces (between filled and empty / boundary cells) are emitted for the
    lateral (XY-plane) surfaces.  Every filled cell also gets front and
    back (Z-plane) faces so the resulting surface is manifold-closed.

    Parameters
    ----------
    grid : np.ndarray
        2D binary array (nx, ny).  1 = filled, 0 = empty.
    filename : str
        Path for the output STL file.
    voxel_size : float
        Edge length of a single voxel (metres).
    origin : tuple of float
        (x, y) world origin of the lower-left corner of the grid.
    z_thickness : float
        Extrusion depth in the Z direction (metres).

    Returns
    -------
    str
        The path to the written STL file.
    """
    nx, ny = grid.shape
    ox, oy = origin
    z0, z1 = 0.0, z_thickness

    triangles = []  # list of (normal, v0, v1, v2)

    def _add_quad(v0, v1, v2, v3, normal):
        """Split a quad into two triangles with the given outward normal."""
        triangles.append((normal, v0, v1, v2))
        triangles.append((normal, v0, v2, v3))

    for i in range(nx):
        for j in range(ny):
            if grid[i, j] == 0:
                continue

            x0 = ox + i * voxel_size
            x1 = ox + (i + 1) * voxel_size
            y0 = oy + j * voxel_size
            y1 = oy + (j + 1) * voxel_size

            # ── lateral faces (only on external boundaries) ──────────────
            # Left  (−x direction)
            if i == 0 or grid[i - 1, j] == 0:
                _add_quad((x0, y0, z0), (x0, y1, z0),
                          (x0, y1, z1), (x0, y0, z1), (-1, 0, 0))

            # Right (+x direction)
            if i == nx - 1 or grid[i + 1, j] == 0:
                _add_quad((x1, y0, z0), (x1, y0, z1),
                          (x1, y1, z1), (x1, y1, z0), (1, 0, 0))

            # Bottom (−y direction)
            if j == 0 or grid[i, j - 1] == 0:
                _add_quad((x0, y0, z0), (x0, y0, z1),
                          (x1, y0, z1), (x1, y0, z0), (0, -1, 0))

            # Top   (+y direction)
            if j == ny - 1 or grid[i, j + 1] == 0:
                _add_quad((x0, y1, z0), (x1, y1, z0),
                          (x1, y1, z1), (x0, y1, z1), (0, 1, 0))

            # ── front / back Z-faces (always emitted) ───────────────────
            # Front (−z)
            _add_quad((x0, y0, z0), (x1, y0, z0),
                      (x1, y1, z0), (x0, y1, z0), (0, 0, -1))

            # Back  (+z)
            _add_quad((x0, y0, z1), (x0, y1, z1),
                      (x1, y1, z1), (x1, y0, z1), (0, 0, 1))

    # ── write ASCII STL ──────────────────────────────────────────────────────
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filename, "w") as fh:
        fh.write("solid voxelShape\n")
        for normal, v0, v1, v2 in triangles:
            fh.write(
                f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
            )
            fh.write("    outer loop\n")
            fh.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            fh.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            fh.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
        fh.write("endsolid voxelShape\n")

    print(f"  [STL] Exported {filename}  ({len(triangles)} triangles, "
          f"{int(np.sum(grid))} filled cells)")
    return filename


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== voxel_generator self-test ===")

    grid = generate_random_grid(seed=42)
    grid = ensure_connected(grid)
    print(f"Random grid  : shape={grid.shape}, filled={np.sum(grid)}")
    voxel_grid_to_stl(grid, "test_random.stl")

    grid2 = generate_airfoil_like_grid()
    print(f"Airfoil grid : shape={grid2.shape}, filled={np.sum(grid2)}")
    voxel_grid_to_stl(grid2, "test_airfoil.stl")

    print("Self-test passed ✓")
