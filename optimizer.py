#!/usr/bin/env python3
"""
optimizer.py — Evolutionary (gradient-free) optimizer for 2D voxel geometries.

Implements a (1+1) evolutionary strategy:
  1. Start with an initial voxel grid (the "parent").
  2. Create a mutant by randomly flipping cells.
  3. Evaluate the mutant via CFD (done externally).
  4. If the mutant improves the objective (downforce), it becomes the new parent.
  5. Repeat for N generations.

The mutation rate is adaptive: large early on (exploration) and small later
(exploitation).

Author: VoxelCFD2D Project
"""

import numpy as np
import os
import json


class VoxelOptimizer:
    """
    (1+1) evolutionary strategy for optimizing 2D voxel grids.

    Parameters
    ----------
    grid_size : tuple of int
        (nx, ny) voxel grid dimensions.
    num_iterations : int
        Total optimisation generations.
    num_mutations : int
        Base number of cells to flip per mutation.
    results_dir : str
        Directory for storing per-generation artefacts.
    """

    def __init__(self, grid_size=(20, 20), num_iterations=20,
                 num_mutations=3, results_dir="results"):
        self.grid_size = grid_size
        self.num_iterations = num_iterations
        self.num_mutations = num_mutations
        self.results_dir = results_dir
        self.generation = 0
        self.rng = np.random.default_rng()

        os.makedirs(results_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Mutation
    # ──────────────────────────────────────────────────────────────────────
    def mutate(self, grid, num_flips=None):
        """
        Create a mutant grid by flipping random cells.

        Parameters
        ----------
        grid : np.ndarray
            Parent voxel grid (2D binary).
        num_flips : int or None
            Number of cells to flip.  Defaults to ``self.num_mutations``.

        Returns
        -------
        np.ndarray
            Mutated copy of the grid.
        """
        if num_flips is None:
            num_flips = self.num_mutations

        mutated = grid.copy()
        nx, ny = mutated.shape

        for _ in range(num_flips):
            i = self.rng.integers(0, nx)
            j = self.rng.integers(0, ny)
            mutated[i, j] = 1 - mutated[i, j]

        self.generation += 1
        return mutated

    def mutate_adaptive(self, grid, iteration, max_iterations):
        """
        Mutation with an adaptive flip count.

        Early iterations flip many cells (exploration); later iterations
        flip fewer (exploitation).

        Parameters
        ----------
        grid : np.ndarray
        iteration : int
            Current iteration index.
        max_iterations : int
            Total planned iterations.

        Returns
        -------
        np.ndarray
        """
        n_flips = self.adaptive_mutation_count(iteration, max_iterations)
        return self.mutate(grid, num_flips=n_flips)

    # ──────────────────────────────────────────────────────────────────────
    # Crossover (optional, for population-based extensions)
    # ──────────────────────────────────────────────────────────────────────
    def crossover(self, grid_a, grid_b):
        """
        Single-point crossover: left half from *grid_a*, right half from *grid_b*.

        Returns
        -------
        np.ndarray
        """
        child = grid_a.copy()
        mid = child.shape[0] // 2
        child[mid:, :] = grid_b[mid:, :]
        return child

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def adaptive_mutation_count(self, iteration, max_iterations):
        """
        Compute the number of flips for the current generation.

        Linearly interpolates from ``max_mutations`` (≈10% of cells) at
        generation 0 down to 1 flip at the final generation.

        Returns
        -------
        int  (≥ 1)
        """
        total_cells = self.grid_size[0] * self.grid_size[1]
        max_mut = max(1, total_cells // 10)
        min_mut = 1
        progress = iteration / max(max_iterations, 1)
        n = int(max_mut * (1.0 - progress) + min_mut * progress)
        return max(1, n)

    def save_state(self, history, filename="optimization_history.json"):
        """Persist the optimisation history to a JSON file."""
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"  [OPT] History saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== optimizer self-test ===")
    opt = VoxelOptimizer(grid_size=(10, 10), num_iterations=5, num_mutations=2)

    parent = np.zeros((10, 10), dtype=np.int32)
    parent[3:7, 3:7] = 1
    print(f"Parent filled cells: {np.sum(parent)}")

    child = opt.mutate(parent)
    print(f"Child  filled cells: {np.sum(child)}")
    print(f"Cells changed      : {np.sum(parent != child)}")

    child2 = opt.mutate_adaptive(parent, iteration=0, max_iterations=10)
    print(f"Adaptive (gen 0)   : {np.sum(parent != child2)} cells changed")

    child3 = opt.mutate_adaptive(parent, iteration=9, max_iterations=10)
    print(f"Adaptive (gen 9)   : {np.sum(parent != child3)} cells changed")

    print("Self-test passed ✓")
