# POROSIM · Pillar 3 — Extractor (MODULAR version)

Visualizes and analyzes the Solver's solutions (Pillar 2): 2D maps of potential
and concentration, field lines, axial profiles, and precipitation zone. Each
analysis is an independent **module**; the extractor discovers them by itself.

> **This copy is the DRY refactor of the extractor.** The Streamlit GUI no
> longer **reimplements** the physics or the plotting: it imports the same
> modules the console mode uses. Editing a module affects the console **and**
> the GUI at once. See [Modular architecture](#modular-architecture-single-source-of-truth).

## Usage modes

```bash
python launch_extractor.py                       # INTERACTIVE: web GUI (Streamlit);
                                                 #   without streamlit falls back to console
python console_backup/extractor.py               # INTERACTIVE: file browser + console menus

# BATCH (no prompts or windows; ALWAYS saves the PNGs in the solution folder)
python launch_extractor.py <solution> <module> --voltage -1.0
python launch_extractor.py <solution> <module> --voltage -1.0 --save-data     # + .txt/.xlsx
python launch_extractor.py <solution> <module> --voltage all                  # full sweep
python launch_extractor.py <solution> solution_summary                        # does not use --voltage
python launch_extractor.py list                  # lists the available modules
```

`--voltage` accepts a value (`-1.0`), a list (`-1.0,0.0,0.5`) or `all`; the
mesh is loaded **only once** for the whole sweep. `<solution>` can be the
`Solutions_*.h5` or its folder.

## File map

| Role | Files |
|---|---|
| **Core** (physics + figures) | `modulos/<category>/*.py` · `modulos/porosim_comun.py` (shared layer) · `solution_summary/` |
| **Web GUI** | `gui_extractor_app.py` (imports the core modules) |
| **Launcher** | `launch_extractor.py` (GUI; batch delegates to the modules) |
| **Console equivalent to the GUI** | `console_backup/extractor.py` (terminal menus; 100% operational) |

- **Interactive**: choose the solution with a graphical browser (remembers the
  last one used), it offers the summary, and you navigate the category → module
  menu.
- **Batch**: runs a module with no menus, via its `preparar()/guardar()`
  contract (Agg backend: never opens windows). The PNG is **always** saved;
  `--save-data` also exports the numerical data of the modules that have it
  (profiles: `.txt`/`.xlsx`; precipitation: node count). Modules with
  `aplica(meta) = False` are skipped (e.g. precipitation on a fully soluble
  salt).

## Input (contract with the Solver)

Each module reads, from the solution folder:

| File | For what |
|---|---|
| `Solutions_<...>.h5` | mesh + U(φ, ln c₊, ln c₋) per voltage |
| `Solutions_<...>_sim.json` | c0, salt, films, channel (z_tip/z_base, R) — metadata |

Without the sibling `_sim.json`, the extractor aborts with a clear message.

## Modules

```
modulos/
├─ porosim_comun.py    shared layer (FEniCS loading + helpers + decoration)
├─ potential_map/     potential.py (2D φ map) · field_lines.py (E = −∇φ)
├─ ion_maps/        ions.py (2D c₊ and c₋) · total_ions.py
├─ precipitation_map/ precipitation.py (precipitation zone, Davies)
└─ axial_profiles/         axis_profile_potential · axis_profile_ions ·
                       section_avg_concentration  (axial profiles r=0)
solution_summary/solution_summary.py   (run summary)
```

Each module implements the **standard contract** (full detail and checklist in
[`modulos/MODULE_CONTRACT.md`](modulos/MODULE_CONTRACT.md)):

| Function | Consumed by | What it does |
|---|---|---|
| `crear_figura(datos, ctx, …)` | console + batch + GUI | arrays → `Figure` (pure) |
| `preparar(ruta, v_label=None, sol=None)` | console + batch | load → `(datos, ctx)` |
| `guardar(datos, ctx, ruta, png, con_datos)` | batch + console prompts | PNG (+ tables) |
| `procesar(ruta)` | console menu | interactive shell (`input`/`show`) |
| `aplica(meta)` *(optional)* | menu + batch | auto-hide / skip |
| `USA_VOLTAJE = False` *(optional)* | batch | global modules (summary) |

To add a new analysis: drop a `.py` with that contract in the appropriate
category; the extractor lists it by itself and batch runs it without touching
the launcher.

## Modular architecture (single source of truth)

Each module is split into **three layers**, so that the physics and the figure
live only once and are consumed by both the console and the GUI:

```
┌───────────────────────────────────────────────────────────────────┐
│  modulos/porosim_comun.py   — shared layer                         │
│    · cargar_meta / contexto_de / info_sal / films_activos_de       │
│    · cargar_malla / espacio_mixto / leer_campos / detectar_voltajes│
│      (FEniCS loading + silenced stderr)                            │
│    · espejo · escala_potencial · guias_canal · bandas_films ·      │
│      rectangulos_films · limpiar_figura   (figure decoration)      │
└───────────────────────────────────────────────────────────────────┘
        ▲                                            ▲
        │ import porosim_comun as pc                 │
┌───────┴──────────────────────────┐        ┌────────┴─────────────────┐
│  modulos/<cat>/<module>.py        │        │  gui_extractor_app.py     │
│    pure PHYSICS (numpy)           │        │    CACHE layer            │
│      calcular_precipitacion, …    │        │    (wraps pc.* with       │
│    crear_figura(datos, ctx, …)    │◀───────│     st.cache_*) +         │
│      → matplotlib Figure          │ import │    export panel           │
│    procesar(ruta_solucion)        │        │    Calls each module's    │
│      console shell (input/        │        │    crear_figura().        │
│      plt.show); uses crear_figura │        │                           │
└───────────────────────────────────┘        └───────────────────────────┘
```

- **`crear_figura(datos, ctx, …)`** is the common signature: it receives NumPy
  arrays in memory (`datos`) and the run context (`ctx = pc.contexto_de(meta,
  stem, v_label)`) and returns a `matplotlib.Figure`. It knows nothing about
  FEniCS, Streamlit or the terminal.
- **`procesar(ruta_solucion)`** (console) loads the data with FEniCS, calls
  `crear_figura` and wraps it in the interactive window + saving prompts.
- **The GUI** caches the data loading (mesh once per `.h5`, fields per voltage)
  and calls the same `crear_figura`. Some modules also expose `dibujar_*(ax, …)`
  primitives that draw on a given `Axes`, so the GUI can **compose** views
  (e.g. potential + concentration stacked in the axial profile) reusing the
  exact plotting.

**To change how an analysis looks or is computed**, edit its module (or
`porosim_comun.py` if it is cross-cutting): the change shows up in the console
and the GUI without touching anything else.

## Outputs

Everything is saved in the **solution folder**. In console mode, each module
opens the interactive window and on closing it offers to save; in batch it saves
directly (PNG always; numerical data with `--save-data`).

The **2D maps** are exported in a **clean** version (only the domain content,
without axes or title) with **the numerical scale bar** (labeled colorbar) so
that the figure is interpretable on its own; the precipitation map (binary)
carries the soluble/precipitates legend. The **1D profiles** are exported as a
full figure (with axes) and their data to `.txt` (and `.xlsx` for the ions
one). `solution_summary` saves the I-V curve, the channel schematic and the
report as `.txt`. Frozen style decisions: see [`PENDIENTES.md`](PENDIENTES.md).

Color conventions: the potential map uses a diverging scale with **white
anchored at 0 V** (the outlet, grounded, looks neutral). Films are marked with a
red border (fixed + charge) or blue (−). All maps are mirrored in r → −r to show
the full pore, with a 1:1 aspect ratio.

## Known limitations

1. **Concentration-map scale anchored to c0.** The color range goes from c0/3
   to 3·c0, with c0 = bulk concentration (single, symmetric). When the
   concentration gradient is implemented in the solver (c_inlet ≠ c_outlet),
   "c0" stops being unique and that range will need to be redefined (see the
   comment in `modulos/ion_maps/ions.py`).
2. **Blocker annotation: pending.** The axial profiles used to mark the band of
   a blocker (solid obstacle in the channel). It was removed from the v1 release
   because the public mesher does not generate blockers yet; it will be re-added
   when the blocker enters the mesher and the solver (see `TODO(blocker)` in the
   `axial_profiles/` modules).
3. **Discretization and integration of the cross-section average profile $\langle c \rangle(z)$.** The computation lives ONCE in `section_avg_concentration.perfil_promedio_seccion()` (called by the console and the GUI) and uses a fixed sampling of `N_Z = 200` uniform axial slices with a radial integral by the trapezoidal rule (`N_R = 35`, linear triangular interpolation). **Planned improvement:** replace the uniform sampling with the real $z$ heights of the mesh nodes in the channel and integrate radially via high-order Gauss-Legendre quadrature, removing fixed constants and taking advantage of the native mesh resolution.
4. **Cross-section average electrostatic potential $\langle \phi \rangle(z)$.** Currently the section-averaged 1D profiles compute only the mean ionic concentrations $\langle c_+ \rangle(z)$ and $\langle c_- \rangle(z)$. **Planned improvement:** add the optional computation of the mean electrostatic potential $\langle \phi \rangle(z) = \frac{2}{R(z)^2} \int_0^{R(z)} \phi(z,r) r dr$ for analysis in 1D area-averaged theories and global energy balance.
