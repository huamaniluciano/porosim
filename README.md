# POROSIM — Nanopore simulation suite (axisymmetric PNP)

[![tests](https://github.com/huamaniluciano/porosim/actions/workflows/tests.yml/badge.svg)](https://github.com/huamaniluciano/porosim/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21402265.svg)](https://doi.org/10.5281/zenodo.21402265)

POROSIM is a comprehensive computational suite designed to model physical chemistry phenomena in nanochannels. It solves the coupled **Poisson-Nernst-Planck (PNP)** equations for electrokinetic ionic transport and electrostatics in axisymmetric geometries using the **Finite Element Method (FEM)**. It allows for the extraction of key physical observables, including Current-Voltage (I-V) curves, 2D spatial maps (e.g., ionic concentration, electric potential), and 1D axial profiles.

The suite provides a complete end-to-end pipeline to simulate ionic transport in various nanopore geometries (e.g., conical, cylindrical, bullet-shaped):
geometry → physics → analysis. Three **pillars** chained by file contracts
(each one consumes what the previous one produces):

```
1_mesher  ──►  2_solver  ──►  3_extractor
 (geometry)      (PNP/FEniCS)    (analysis and figures)
      │               │                │
      ▼               ▼                ▼
 RESULTS/meshes   RESULTS/solutions   figures/tables in
 (.xdmf + limites)   (.h5 + _sim.json + I-V)  each solution's folder
```

## Quickstart

### 1. Install

POROSIM runs on Linux with a conda environment (it uses the legacy FEniCS
2019.1.0 stack). From the repository root:

```bash
# On a minimal or headless Linux, install the system libraries gmsh links
# against (already present on most desktops; Debian/Ubuntu shown):
sudo apt-get install -y libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1

# Create and activate the environment (installs FEniCS, gmsh, ...; this can
# take several minutes):
conda env create -f environment.yml
conda activate porosim
```

To confirm the install, run the test suite (under a minute):

```bash
python -m pytest        # expect: 31 passed
```

### 2. Run the full pipeline on the small example

This chains the three pillars on a small conical pore (KCl 100 mM). The whole
run takes well under a minute.

```bash
# 1. Mesh — build the geometry (~5 s)
cd 1_mesher
python launch_mallador.py examples/geom_conico_chico.json
#   -> 1_mesher/mallas/conico_chico/   (<name>_domain.xdmf, _facets.xdmf, _limits.json)

# 2. Solve — the PNP physics on that mesh (~5 s)
cd ../2_solver
python solver.py examples/params_conico_chico.json --mesh ../1_mesher/mallas/conico_chico
#   -> RESULTS/solutions/conico_chico_KCl_100.0mM/   (Solutions_*.h5 + I-V table)

# 3. Extract — a 2D potential map at +0.4 V, plus its numerical data
cd ../3_extractor
python launch_extractor.py \
    ../RESULTS/solutions/conico_chico_KCl_100.0mM/Solutions_conico_chico_KCl_100.0mM.h5 \
    potential --voltage 0.4 --save-data
#   -> the PNG lands next to the solution, in that same folder
```

You now have a mesh, an I-V curve, and a potential-map figure. For the
interactive GUIs instead of the batch commands, run any launcher without
arguments (see below).

## How to use it

**Web portal (recommended)** — a single multipage app with the GUIs of the
three pillars:

```bash
python launch_porosim.py        # opens the browser (Streamlit)
```

**Per pillar** — each pillar works on its own, with its own GUI or via
console/batch:

| Pillar | Web GUI | Batch (iterable) | Interactive console |
|---|---|---|---|
| 1_mesher | `python launch_mallador.py` | `python launch_mallador.py geom.json` | — (the GUI is the only interactive mode) |
| 2_solver | `python launch_solver.py` | `python solver.py params.json` | `python solver.py` |
| 3_extractor | `python launch_extractor.py` | `python console_backup/extractor.py <sol.h5> <module> --voltage V` | `python console_backup/extractor.py` |

**Naming rule**: everything runnable starts with `launch_` (one launcher per
pillar + the portal); the `gui_*.py` are the web interface those launchers open
(not run by hand); the rest is each pillar's core.

The console versions equivalent to the GUI live in `console_backup/` inside
each pillar (where they exist): they are the functional reference the GUI
replicates, and remain 100% operational.

## Structure

```
POROSIM/
├─ launch_porosim.py        Multipage web portal (launches the 3 GUIs; without
│                           Streamlit falls back to a console menu)
├─ portada_porosim.py       Portal home page
├─ 1_mesher/              Geometry and meshing (gmsh + meshio; NO FEniCS)
│  ├─ capa1..capa4_*.py     Core: model → loops → gmsh → mesh
│  ├─ gui_*.py              Web GUI
│  ├─ launch_mallador.py    Launcher (GUI / batch)
│  └─ examples/             example geom.json
├─ 2_solver/                PNP physics (FEniCS/dolfin 2019.1.0)
│  ├─ motor_pnp.py          Core: the paper's physics (resolver(cfg))
│  ├─ constantes.py · sales.json · solver.py (batch + console)
│  ├─ gui_*.py              Web GUI
│  ├─ launch_solver.py      GUI launcher (batch delegates to solver.py)
│  ├─ console_backup/preguntas.py   Console prompts the GUI mimics
│  └─ examples/             example params.json
├─ 3_extractor/             Solution analysis (FEniCS to read the .h5)
│  ├─ modulos/              Core: one .py per analysis + porosim_comun.py
│  ├─ gui_extractor_app.py  Web GUI (imports the modules: single source)
│  ├─ launch_extractor.py   GUI launcher
│  ├─ console_backup/extractor.py  Console menus equivalent to the GUI
│  └─ resumen_solucion/     Text report of a run
└─ RESULTS/              Shared I/O (meshes, equilibria, solutions)
```

## Dependencies

- **Pillar 1**: `gmsh`, `meshio`, `numpy`, `matplotlib` (no FEniCS).
- **Pillars 2 and 3**: FEniCS/dolfin 2019.1.0 + `numpy` + `matplotlib`.
- **GUIs / portal**: `streamlit` (optional: without it, pillars 2 and 3 fall
  back to their console mode). `pandas`/`openpyxl` optional to export tables.

Each pillar has its own `README.md` with the complete guide to its modes,
input/output contracts and known limitations.

## Contributing

Bug reports, questions, and contributions are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to report a bug, ask for help, or
submit a pull request. All participation is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).
