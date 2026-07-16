# POROSIM — Nanopore simulation suite (axisymmetric PNP)

Complete pipeline to simulate ionic transport in conical nanopores:
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
