# VoxelCFD2D — Voxel-Based 2D CFD Shape Optimisation

Optimise a 2D voxel geometry within a bounding box to achieve **maximum
downforce**, using OpenFOAM 13 for CFD and Python for orchestration.

---

## Project Goal

| Item | Detail |
|------|--------|
| **Geometry** | 2D binary voxel grid (filled / empty cells) |
| **CFD solver** | `pisoFoam` (transient, incompressible, laminar) |
| **Meshing** | `blockMesh` + `snappyHexMesh -overwrite` |
| **Objective** | Maximise downforce (−F_y on the body) |
| **Optimiser** | (1+1) evolutionary strategy with adaptive mutation |

---

## Directory Layout

```
shapeOptimizer/
├── README.md                       ← this file
├── requirements.txt                ← Python dependencies
├── .gitignore
│
├── voxel_generator.py              ← grid generation & STL export
├── optimizer.py                    ← evolutionary mutation logic
├── visualization.py                ← plotting (grids, history, STL)
├── run_project.py                  ← main entry point (full loop)
│
├── mesh_setup/
│   ├── blockMeshDict               ← base Cartesian mesh (150×75×1)
│   └── snappyHexMeshDict           ← castellated mesh around STL
│
├── solver_setup/
│   ├── controlDict                 ← pisoFoam settings + force function
│   ├── fvSolution                  ← PISO algorithm parameters
│   └── fvSchemes                   ← discretisation schemes
│
├── boundary_conditions/
│   ├── U                           ← velocity BCs (inlet/outlet/walls)
│   └── p                           ← pressure BCs
│
├── constant_properties/
│   ├── transportProperties         ← kinematic viscosity (nu)
│   └── turbulenceProperties        ← laminar flow declaration
│
├── results/                        ← output artefacts (plots, grids, JSON)
│   └── .gitkeep
│
└── cfd_case/                       ← (runtime) OpenFOAM case — gitignored
```

---

## Environment Setup

### 1. OpenFOAM 13

| Item | Value |
|------|-------|
| Version | OpenFOAM 13 (Foundation) |
| Install path | `/opt/openfoam13/` |
| Tutorials | `/opt/openfoam13/tutorials/` |

```bash
# Always source before running:
source /opt/openfoam13/etc/bashrc
```

### 2. Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. WSL2 Notes

* Work in the Linux home directory (`~/projects/...`) for best I/O performance.
* Deactivate Conda before sourcing OpenFOAM:
  ```bash
  conda deactivate
  source /opt/openfoam13/etc/bashrc
  ```

---

## How to Run

```bash
# 1. Prepare environment
conda deactivate
source /opt/openfoam13/etc/bashrc

# 2. (Optional) activate Python venv
source .venv/bin/activate

# 3. Run the full optimisation loop
python3 run_project.py
```

The script will:

1. Generate an initial airfoil-like voxel grid.
2. Export the geometry as an ASCII STL.
3. Assemble an OpenFOAM case (`cfd_case/`).
4. Run `blockMesh` → `snappyHexMesh -overwrite` → `pisoFoam`.
5. Parse forces from `postProcessing/forces/`.
6. Compare downforce with the best-so-far; keep if improved.
7. Mutate the voxel grid and repeat.
8. Save plots, `.npy` grids, and a JSON history to `results/`.

---

## Configuration

Edit the constants at the top of `run_project.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRID_SIZE` | `(20, 20)` | Voxel grid resolution |
| `NUM_ITERATIONS` | `10` | Optimisation generations |
| `NUM_MUTATIONS` | `3` | Base cells flipped per mutation |
| `USE_ADAPTIVE_MUTATION` | `True` | Taper mutation rate over time |

Flow parameters live in the OpenFOAM dictionaries:

| File | Key parameter |
|------|---------------|
| `constant_properties/transportProperties` | `nu` (viscosity) |
| `boundary_conditions/U` | inlet velocity |
| `solver_setup/controlDict` | `endTime`, `deltaT` |

---

## Output / Results

After a run, `results/` contains:

| File | Description |
|------|-------------|
| `voxel_grid_NNN.npy` | NumPy array of each generation's grid |
| `voxel_NNN.png` | Colour-coded grid plot per iteration |
| `best_grid.npy` / `best_grid.png` | Best geometry found |
| `best_shape.stl` | STL of the best geometry |
| `best_stl_preview.png` | 3D wireframe of the best STL |
| `optimization_history.json` | Full history (downforce, fill count) |
| `optimization_history.png` | Downforce + complexity over iterations |
| `comparison.png` | Side-by-side initial vs. best |

---

## Technical Details

### Voxel → STL

Each filled cell in the 20×20 grid is a `0.025 m × 0.025 m` box,
extruded `0.01 m` in Z.  Only **external faces** are emitted
(lateral faces between filled and empty neighbours), plus front / back
Z-faces for every filled cell — producing a watertight surface mesh.

### Meshing Strategy

1. **blockMesh** creates a uniform Cartesian base mesh
   (150 × 75 × 1 cells, ~0.02 m cell size) with `empty` front/back
   patches for the 2D constraint.
2. **snappyHexMesh** runs in *castellated-only* mode (`snap false`,
   `addLayers false`, refinement level `(0 0)`) to remove cells that
   fall inside the STL body.  No isotropic refinement is applied, so
   the single Z-layer is preserved.

### Forces

The `controlDict` includes a `forces` function object that writes
pressure + viscous forces on the `voxelShape` patch to
`postProcessing/forces/0/force.dat`.  The Python parser reads the
y-component and returns `downforce = −F_y`.

### Optimiser

A simple **(1+1) ES**: one parent, one offspring.  The offspring is
kept if it matches or beats the parent's downforce.  The mutation
rate starts high (≈10 % of cells) and decays linearly to 1 cell at
the final generation.

---

## Extending to 3D

1. Remove the `empty` patches; make front/back walls or symmetry.
2. Increase Z resolution in `blockMeshDict`.
3. Enable snapping and refinement in `snappyHexMeshDict`.
4. Switch to a 3D voxel grid in `voxel_generator.py`.
5. Consider using a turbulence model (`kOmegaSST`, etc.).

---

## License

MIT — see individual file headers.
