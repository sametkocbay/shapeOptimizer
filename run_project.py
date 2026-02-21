#!/usr/bin/env python3
"""
run_project.py — Main orchestration for the 2D Voxel-Based CFD Optimisation.

Pipeline executed every iteration:
  1. Generate / mutate a 2D voxel grid.
  2. Export the grid to an ASCII STL.
  3. Assemble a fresh OpenFOAM case directory (0/, constant/, system/).
  4. Run  blockMesh  → base Cartesian mesh.
  5. Run  snappyHexMesh -overwrite  → carve out the solid body.
  6. Run  pisoFoam  → transient CFD solver.
  7. Parse force output → extract downforce (−Fy).
  8. Compare with best-so-far; keep or discard the mutation.
  9. Save artefacts (grid .npy, grid plot, history JSON).
  10. Repeat from step 1.

Usage
-----
    conda deactivate                          # avoid Conda / OF clashes
    source /opt/openfoam13/etc/bashrc         # load OpenFOAM 13
    python3 run_project.py                    # run full optimisation

Author: VoxelCFD2D Project
"""

import os
import sys
import shutil
import subprocess
import json
import time

import numpy as np

from voxel_generator import (
    generate_random_grid,
    generate_airfoil_like_grid,
    generate_wedge_grid,
    ensure_connected,
    voxel_grid_to_stl,
    DEFAULT_VOXEL_SIZE,
    DEFAULT_ORIGIN,
    DEFAULT_Z_THICKNESS,
)
from optimizer import VoxelOptimizer
from visualization import (
    plot_voxel_grid,
    plot_optimization_history,
    plot_stl_wireframe,
    plot_comparison,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.join(PROJECT_DIR, "cfd_case")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

GRID_SIZE = (20, 20)           # voxel grid resolution (nx, ny)
NUM_ITERATIONS = 10            # optimisation generations (increase for better results)
NUM_MUTATIONS = 3              # base cells flipped per mutation
USE_ADAPTIVE_MUTATION = True   # decrease mutation rate over time

OPENFOAM_SOURCE = "source /opt/openfoam13/etc/bashrc"

# ══════════════════════════════════════════════════════════════════════════════
# OpenFOAM case assembly
# ══════════════════════════════════════════════════════════════════════════════

def setup_openfoam_case(grid, case_dir):
    """
    Build a complete OpenFOAM case directory from scratch.

    Cleans any prior case, creates the directory tree, exports the STL,
    and copies all dictionary / BC / property files.

    Parameters
    ----------
    grid : np.ndarray
        2D binary voxel grid (nx, ny).
    case_dir : str
        Absolute path to the case directory.
    """
    # wipe previous run
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

    # create skeleton
    for sub in ("0", "constant/triSurface", "system"):
        os.makedirs(os.path.join(case_dir, sub), exist_ok=True)

    # 1. STL geometry
    stl_path = os.path.join(case_dir, "constant", "triSurface", "voxelShape.stl")
    voxel_grid_to_stl(grid, stl_path)

    # 2. system/ dictionaries
    _copy_dicts(case_dir)


def _copy_dicts(case_dir):
    """Copy all OpenFOAM dictionary files into *case_dir*."""
    sys_dst = os.path.join(case_dir, "system")
    zero_dst = os.path.join(case_dir, "0")
    const_dst = os.path.join(case_dir, "constant")

    # system/ ← solver_setup + mesh_setup
    for src_dir, names in [
        (os.path.join(PROJECT_DIR, "solver_setup"),
         ["controlDict", "fvSolution", "fvSchemes"]),
        (os.path.join(PROJECT_DIR, "mesh_setup"),
         ["blockMeshDict", "snappyHexMeshDict"]),
    ]:
        for name in names:
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(sys_dst, name))

    # 0/ ← boundary_conditions
    bc_dir = os.path.join(PROJECT_DIR, "boundary_conditions")
    for name in ("U", "p"):
        src = os.path.join(bc_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(zero_dst, name))

    # constant/ ← constant_properties
    prop_dir = os.path.join(PROJECT_DIR, "constant_properties")
    for name in ("transportProperties", "turbulenceProperties"):
        src = os.path.join(prop_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(const_dst, name))


# ══════════════════════════════════════════════════════════════════════════════
# OpenFOAM solver execution
# ══════════════════════════════════════════════════════════════════════════════

def run_openfoam(case_dir):
    """
    Execute the meshing + solver pipeline via subprocess calls.

    Sequence: blockMesh → snappyHexMesh -overwrite → pisoFoam

    Returns
    -------
    bool
        True if every command exited with code 0.
    """
    commands = [
        ("blockMesh", f"cd {case_dir} && blockMesh"),
        ("snappyHexMesh", f"cd {case_dir} && snappyHexMesh -overwrite"),
        ("pisoFoam", f"cd {case_dir} && pisoFoam"),
    ]

    for label, cmd in commands:
        full_cmd = f"bash -c '{OPENFOAM_SOURCE} && {cmd}'"
        print(f"\n  >>> {label}")
        t0 = time.time()

        result = subprocess.run(
            full_cmd, shell=True,
            capture_output=True, text=True,
            timeout=600,  # 10 min safety timeout
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  ✗ {label} FAILED  ({elapsed:.1f} s)")
            # print truncated stderr for diagnostics
            err = result.stderr.strip()
            if err:
                for line in err.split("\n")[-15:]:
                    print(f"    {line}")
            return False

        print(f"  ✓ {label} OK  ({elapsed:.1f} s)")
        # show last few stdout lines as a sanity check
        for line in result.stdout.strip().split("\n")[-3:]:
            print(f"    {line}")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Force extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_forces(case_dir):
    """
    Parse the OpenFOAM forces function-object output and return the
    time-averaged downforce in Newtons.

    Downforce = −Fy  (negative y-force → positive downforce).

    The parser handles the OpenFOAM Foundation v13 output format::

        # Time  ((px py pz) (vx vy vz) (porx pory porz))

    Returns
    -------
    float
        Average downforce over the last 50 time steps (or fewer if
        the simulation is shorter).
    """
    pp_dir = os.path.join(case_dir, "postProcessing")
    if not os.path.isdir(pp_dir):
        print("  [FORCE] postProcessing/ not found → downforce = 0")
        return 0.0

    # Walk through possible paths
    force_file = None
    for root, _dirs, files in os.walk(pp_dir):
        for fname in files:
            if fname in ("force.dat", "forces.dat", "force_0.dat"):
                force_file = os.path.join(root, fname)
                break
        if force_file:
            break

    if force_file is None:
        print("  [FORCE] No force.dat found → downforce = 0")
        return 0.0

    fy_values = []
    with open(force_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                # Replace parentheses with spaces so we can split by whitespace
                tokens = line.replace("(", " ").replace(")", " ").split()
                # tokens[0] = time
                # tokens[1..3] = pressure force  (px py pz)
                # tokens[4..6] = viscous force   (vx vy vz)
                fy_pressure = float(tokens[2])
                fy_viscous  = float(tokens[5])
                fy_values.append(fy_pressure + fy_viscous)
            except (ValueError, IndexError):
                continue

    if not fy_values:
        print("  [FORCE] Could not parse any force data → downforce = 0")
        return 0.0

    # Average the last N samples for a more stable metric
    n_avg = min(50, len(fy_values))
    avg_fy = np.mean(fy_values[-n_avg:])

    # Convention: downforce is positive when Fy is negative
    downforce = -avg_fy
    return downforce


# ══════════════════════════════════════════════════════════════════════════════
# Main optimisation loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║        Voxel-Based 2D CFD Optimisation (pisoFoam)          ║
    ║                   Maximising Downforce                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Initialise optimiser ─────────────────────────────────────────────
    optimizer = VoxelOptimizer(
        grid_size=GRID_SIZE,
        num_iterations=NUM_ITERATIONS,
        num_mutations=NUM_MUTATIONS,
        results_dir=RESULTS_DIR,
    )

    # ── Generate initial geometry ────────────────────────────────────────
    grid = generate_airfoil_like_grid(GRID_SIZE)
    grid = ensure_connected(grid)
    initial_grid = grid.copy()

    best_grid = grid.copy()
    best_downforce = -np.inf
    initial_downforce = 0.0
    history = []

    print(f"  Grid size       : {GRID_SIZE[0]}×{GRID_SIZE[1]}")
    print(f"  Iterations      : {NUM_ITERATIONS}")
    print(f"  Adaptive mutate : {USE_ADAPTIVE_MUTATION}")
    print(f"  Results dir     : {RESULTS_DIR}")

    # ── Optimisation loop ────────────────────────────────────────────────
    for it in range(NUM_ITERATIONS):
        print(f"\n{'═' * 60}")
        print(f"  ITERATION {it + 1} / {NUM_ITERATIONS}")
        print(f"{'═' * 60}")

        # 1. Assemble OpenFOAM case
        setup_openfoam_case(grid, CASE_DIR)

        # 2. Run CFD pipeline
        success = run_openfoam(CASE_DIR)

        # 3. Extract downforce
        if success:
            downforce = extract_forces(CASE_DIR)
        else:
            print("  CFD pipeline failed — assigning downforce = 0")
            downforce = 0.0

        if it == 0:
            initial_downforce = downforce

        print(f"\n  Downforce = {downforce:+.6f} N")

        # 4. Selection: keep if better
        if downforce > best_downforce:
            best_downforce = downforce
            best_grid = grid.copy()
            print(f"  ★ NEW BEST  (downforce = {best_downforce:+.6f} N)")

        # 5. Record history
        record = {
            "iteration": it,
            "downforce": float(downforce),
            "best_downforce": float(best_downforce),
            "filled_cells": int(np.sum(grid)),
        }
        history.append(record)

        # 6. Save artefacts for this iteration
        np.save(os.path.join(RESULTS_DIR, f"voxel_grid_{it:03d}.npy"), grid)
        plot_voxel_grid(
            grid, iteration=it, downforce=downforce,
            save_path=os.path.join(RESULTS_DIR, f"voxel_{it:03d}.png"),
        )

        # 7. Mutate for the next generation
        if USE_ADAPTIVE_MUTATION:
            grid = optimizer.mutate_adaptive(best_grid, it, NUM_ITERATIONS)
        else:
            grid = optimizer.mutate(best_grid)
        grid = ensure_connected(grid)

    # ══════════════════════════════════════════════════════════════════════
    # Final report
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  OPTIMISATION COMPLETE")
    print(f"  Best downforce : {best_downforce:+.6f} N")
    print(f"  Best fill      : {int(np.sum(best_grid))} / {best_grid.size} cells")
    print(f"{'═' * 60}\n")

    # Save best grid
    np.save(os.path.join(RESULTS_DIR, "best_grid.npy"), best_grid)
    plot_voxel_grid(
        best_grid, iteration="BEST", downforce=best_downforce,
        save_path=os.path.join(RESULTS_DIR, "best_grid.png"),
    )

    # History plot
    plot_optimization_history(
        history,
        save_path=os.path.join(RESULTS_DIR, "optimization_history.png"),
    )

    # Comparison: initial vs best
    plot_comparison(
        initial_grid, best_grid,
        initial_downforce, best_downforce,
        save_path=os.path.join(RESULTS_DIR, "comparison.png"),
    )

    # STL wireframe of best geometry
    stl_best = os.path.join(RESULTS_DIR, "best_shape.stl")
    voxel_grid_to_stl(best_grid, stl_best)
    plot_stl_wireframe(
        stl_best,
        save_path=os.path.join(RESULTS_DIR, "best_stl_preview.png"),
    )

    # Persist history as JSON
    optimizer.save_state(history, "optimization_history.json")

    print(f"  All artefacts saved to  {RESULTS_DIR}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
