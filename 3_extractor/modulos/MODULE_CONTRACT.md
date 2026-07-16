# Extractor module contract (Pillar 3)

How to write a new module so it works IDENTICALLY in the 3 modes: console
(menu), batch (`launch_extractor.py`) and GUI (`gui_extractor_app.py`).

## Location

```
modulos/<category>/<name>.py      → the extractor discovers it on its own.
```
New categories = new folders; nothing to register anywhere.

## Boilerplate (import the shared layer)

```python
import sys, pathlib
_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc
```

## The 4 contract functions

### 1) `crear_figura(datos, ctx, **estetica) -> matplotlib.figure.Figure`
**[REQUIRED — consumed by EVERYONE]**
- `datos`: dict of in-memory NumPy arrays (document the keys in the module
  docstring). NO FEniCS, NO paths, NO `input()`, NO `show()`.
- `ctx`: `pc.contexto_de(meta, stem, v_label)` — brings `c0`, salt, `T_K`,
  `V_T`, channel (`z_tip`/`z_base`/`R_*`), films, sigma. With `v_label`/`v_num`
  already set.
- Aesthetic kwargs with sensible defaults (`leyenda=True`, `titulo=True`,
  `figsize=...`, etc.). The GUI uses them for the variants.
- If the module composes onto a foreign `Axes` (the GUI's stacked views), also
  expose a `dibujar_<something>(ax, datos, ctx, ...)` primitive.

### 2) `preparar(ruta_solucion, v_label=None, sol=None) -> (datos, ctx) | None`
**[REQUIRED — consumed by console and batch]**
- `v_label=None` → interactive mode: `pc.elegir_voltaje` asks on the console.
- `v_label="-1.0"` (str or float) → batch mode: validates without asking.
- `sol` = dict from `pc.cargar_solucion()` ALREADY LOADED. Batch loads it ONCE
  and passes it here for each voltage (the mesh is not reloaded). With
  `sol=None` it loads internally (console: one module, one voltage).
- Delegate the common trunk to `pc.preparar_comun(ruta, TITULO, v_label, sol)`
  → `(sol, ctx, campos) | None`. Then transform `campos` + `sol["dm"]` into the
  module's own `datos` dict. DO NOT duplicate the mesh/voltage loading.
- Returns `None` (with the reason already printed) if something fails or does
  not apply.

### 3) `guardar(datos, ctx, ruta_solucion, png=True, con_datos=False) -> [Path]`
**[REQUIRED — consumed by batch and the console prompts]**
- `png=True` → saves the module's image(s) in the SOLUTION FOLDER. 2D maps:
  CLEAN version (`pc.guardar_figura(..., limpio=True)`: no axes/title, colorbar
  yes — publication-ready). 1D profiles and multi-panel figures: full figure
  with axes.
- `con_datos=True` → also exports the tabular data with `pc.exportar_txt` (and
  `pc.exportar_xlsx` if applicable). If the module has no tabular data, it STILL
  ACCEPTS the kwarg and ignores it (contract uniformity).
- NO `input()`, NO `plt.show()`. Use the `pc` helpers (dpi 300, bbox tight,
  automatic close, prints "✓ Saved:").
- File names (existing convention, don't invent another):
    - maps:     `{stem}_<suffix>_{V}V.png`   (e.g. `{stem}_potential_map_-1.00V.png`)
    - profiles: `<analysis>_{V}V.png / .txt / .xlsx`   (e.g. `ion_profile_-1.00V.png`)
- Returns the list of saved Paths.

### 4) `procesar(ruta_solucion)`
**[REQUIRED — console shell, the ONLY place with `input()`/`plt.show()`]**
```python
prep = preparar(ruta_solucion)          # interactive (asks for voltage)
if prep is None: return
datos, ctx = prep
crear_figura(datos, ctx); plt.show()
... input("Save...?") → guardar(datos, ctx, ruta, png=..., con_datos=...)
```

## Optional

**`aplica(meta) -> bool`**
Does the module make sense for this solution? (e.g. precipitation only with
sparingly soluble salts). The menu hides it; BATCH SKIPS it. Without `aplica()`
→ always True.

**`USA_VOLTAJE = False`**
Global modules that analyze the full sweep (`solution_summary`): `preparar()`
ignores `v_label`/`sol` and batch does not require `--voltage`.

## Who calls what

- **console** (`launch_extractor` / `console_backup` menu):
  `procesar(ruta)`
- **batch** (`launch_extractor.py`, Agg backend, no stdin):
  ```python
  sol = pc.cargar_solucion(ruta)                       # once
  for each voltage: preparar(ruta, v, sol=sol) → guardar(..., con_datos)
  ```
  NEVER calls `procesar()`. A module without `preparar`/`guardar` → clear error.
- **GUI** (`gui_extractor_app.py`):
  `crear_figura(...)` / `dibujar_*(...)` with its own data cache.

## Golden rules

- The physics and the figure live ONCE (consumed by all 3 modes).
- `input()` ONLY in `procesar()` (and in `pc.elegir_voltaje`, interactive branch).
- `plt.show()` ONLY in `procesar()`.
- `savefig` ONLY via `guardar()` (using the `pc` helpers).
- Everything cross-cutting (loading, mirroring, scales, films, standard saving)
  goes to `porosim_comun.py`, not copied between modules.

## Checklist for a new module

- [ ] `.py` in `modulos/<category>/` with the `pc` boilerplate
- [ ] pure `crear_figura(datos, ctx, ...)`
- [ ] `preparar(ruta, v_label=None, sol=None)` via `pc.preparar_comun`
- [ ] `guardar(datos, ctx, ruta, png=True, con_datos=False)` via `pc` helpers
- [ ] `procesar(ruta)` = preparar + show + prompts + guardar
- [ ] (if applicable) `aplica(meta)` / `USA_VOLTAJE = False`
- [ ] test it in the 3 modes:
    - menu:  `python launch_extractor.py`   (console fallback)
    - batch: `python launch_extractor.py <sol> <module> --voltage -1.0`
    - GUI:   add it to `gui_extractor_app.py` if it has its own tab
