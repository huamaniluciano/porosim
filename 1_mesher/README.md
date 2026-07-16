# POROSIM · Pillar 1 — Mesher

Generates the 2D axisymmetric mesh (channel + reservoirs) consumed by the
Solver (Pillar 2). Output per mesh: `<name>_domain.xdmf` + `<name>_facets.xdmf` +
`<name>_limits.json` (physical tags), plus `.msh` and an inspection `.png`.

Dependencies: `gmsh`, `meshio`, `numpy`, `matplotlib` (+ `streamlit` only for
the GUI mode). Does **not** require FEniCS.

## Two usage modes

```bash
python launch_mallador.py                   # INTERACTIVE: web GUI (Streamlit)
python launch_mallador.py geom.json         # BATCH: no questions, iterable
```

## File map

| Role | Files |
|---|---|
| **Core** (meshing pipeline) | `capa1_modelo.py` (parameters and profile) → `capa2_loops.py` (contours) → `capa3_gmsh.py` (geometry) → `capa4_malla.py` (mesh + tags) |
| **Web GUI** | `gui_app.py` · `gui_dibujo.py` · `gui_a_params.py` |
| **Launcher** | `launch_mallador.py` (no args → GUI; with geom.json → batch) |
| **Examples** | `examples/geom_*.json` |

This pillar has no interactive console version: the interactive mode IS the
GUI, and the reproducible mode is batch with `geom.json`.

Both modes call the same `mallar()` function (capa4_malla), so for the same
parameters the mesh is identical. The `geom.json` (see `examples/`) has three
fields: `name`, `output` (folder) and `params` — the latter maps 1:1 to the
`Params` dataclass in `capa1_modelo.py` (SI units, in meters).

## geom.json guide

`python launch_mallador.py help` prints this guide with the current defaults
(read from the code, not from this file). General structure:

```json
{
  "name": "my_mesh",                     // required: base name of the files
  "output": "meshes/my_mesh",            // optional: folder (relative to cwd or absolute)
  "params": { ... }                      // required: the geometry (fields below)
}
```

`params` fields — **all optional** (whatever is omitted takes its default);
an unknown name aborts with an error (protects against typos). SI units in
meters: `20e-9` = 20 nm, `2e-6` = 2 µm.

| Field | Default | What it is |
|---|---|---|
| `L_pore` | `50e-9` | channel length |
| `D_tip` / `D_base` | `10e-9` / `50e-9` | mouth diameters (tip = the narrow one) |
| `L_res` / `R_res` | `500e-9` / `400e-9` | length and radius of the reservoirs |
| `L_charge` | `5e-9` | width of the chargeable ring on each face |
| `L_far` | `45e-9` | width of the transition zone past the ring |
| `include_film_tip` | `false` | film attached to the tip mouth |
| `delta_film_tip` | `10e-9` | thickness of that film |
| `include_film_base` / `delta_film_base` | `false` / `10e-9` | same on the base side |
| `channel_type` | `"conical"` | `"cylinder"` \| `"conical"` \| `"bullet"` |
| `h_param` | `50e-9` | bullet scale (only if `channel_type="bullet"`) |
| `N_PTS_WALL` | `200` | points of the wall spline (leave default) |

**Bullet channel** (exponential profile): add the two keys

```json
"params": { ..., "channel_type": "bullet", "h_param": 2.0e-6 }
```

R(x) = R_base − (R_base − R_tip)·exp(−x/h_param), with x from the tip mouth.
small `h` ⇒ the channel opens up fast near the tip; large `h` ⇒ smooth
transition. Runnable example: `examples/geom_bullet_chico.json` (~1 min); the
paper-geometry version: `examples/geom_bullet_paper.json` (heavy).

**Cylinder**: `"channel_type": "cylinder"` with `D_tip == D_base` (internally
it is the conical one with both diameters equal).

## Architecture (4-layer pipeline + GUI)

```
gui_app.py ──► gui_dibujo.py            (GUI: sliders + live drawing)
     │
gui_a_params.py                         (GUI state → Params)
     │                    launch_mallador.py ── geom.json → Params
     ▼                          ▼
capa1_modelo.py   MODEL. Params + declarative derivation of the topology
     ▼            (stations/slabs → points, lines, tags). No Gmsh.
capa2_loops.py    LOOPS. Oriented curve loops + closure and shared-edge
     ▼            verification, in pure Python. No Gmsh.
capa3_gmsh.py     EMISSION. Translates the model to Gmsh (points, splines,
     ▼            loops, surfaces, physical groups).
capa4_malla.py    MESH. Multi-scale refinement (0.2 nm at the wall → µm in
                  bulk) and export to the solver contract (xdmf + limites.json).
```

Central rule of layer 1: *a line exists on a band ⟺ the region on the left
≠ the region on the right*. From that the whole topology is derived for the
four film configurations (none/tip/base/both) with no special cases.

The wall-profile formula lives in **one single place**: `perfil_radio()`
(`capa1_modelo.py`). It is used by the layers (via `R_canal`) and by the GUI
drawing, so the drawing and the mesh cannot diverge. Documented exception: layer
4 repeats the formula as a Gmsh MathEval *string* (not callable); if the profile
changes, both places must be updated.

## Self-tests (each layer runs standalone)

```bash
python capa1_modelo.py     # topology dump + matplotlib preview (4 configs)
python capa2_loops.py      # verifies the loops of the 4 configs (no Gmsh, fast)
python capa3_gmsh.py       # emits the 4 configs to Gmsh and reports OK/FAIL;
                           #   at the end it offers inspection in the Gmsh GUI
python capa4_malla.py      # meshes the 4 full demo configs (slow)
python gui_dibujo.py       # generates PNGs of example design states
```

To validate a change: `capa2` and `capa3` green + regenerate a mesh from
`examples/` and compare the triangle count (or the md5 of the `.msh`).

## Physical tags (contract with the Solver)

| Tag | Name | | Tag | Name |
|----|----|----|----|----|
| 1 | AXIS | | 6 | CHARGE_ZONE_BASE |
| 2 | WALL | | 7 | FILM_TIP_INTERFACE |
| 3 | INLET | | 8 | FILM_BASE_INTERFACE |
| 4 | OUTLET | | 10 | DOMAIN_FLUID |
| 5 | CHARGE_ZONE_TIP | | 11/12 | DOMAIN_FILM_TIP/BASE |

## Current limitations

- The blocker (partial occlusion, deposit/precipitate type) is **not** in this
  engine yet; it will be added as a sub-field of `geom.json` (it cuts the wall
  and adds new tags, requiring layers 1–2 to be extended).
- The GUI writes to `../RESULTS/meshes/<name>/`; batch mode writes wherever
  the JSON `output` field says.
- `N_PTS_WALL` (points of the wall spline) has a fixed default (200) that does
  **not** scale with the geometry — deriving it from `L_pore`/`h_param` (as the
  refiner's `Sampling` already does, layer 4) is pending. Meanwhile, for very
  long bullets with small `h`: raise it by hand in the `geom.json` if the wall
  looks faceted near the tip (see the TODO in `capa1_modelo.py`).
