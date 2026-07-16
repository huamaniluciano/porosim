# Extractor pending items / FROZEN decisions (to revisit later)

Conscious v1 choices that were deliberately frozen (2026-07-14). They are not
bugs: they are decisions. Each one says where it lives in the code so it can be
changed in ONE single place when the time comes.

## 1. Supersaturation factor = 2 (the paper's) — FROZEN

Precipitates where `Q_act > 2·Ksp` (S ≈ 1.4). It is the factor used in the
paper figures.

- **Where it lives**: `modulos/precipitation_map/precipitation.py` →
  constant `FACTOR_PAPER = 2.0`. It is the default of the batch (`guardar()`),
  of `crear_figura()`, of the console TextBox and of the GUI slider.
- **To change it**: touch ONLY `FACTOR_PAPER`. For a one-off batch run another
  factor can be passed via `guardar(..., factor=X)` (deliberately not exposed on
  the CLI: the paper uses 2).

## 2. Activity model: Davies with a cap at I = 0.5 M — DO NOT MOVE

`γ±` is computed with the Davies equation (A = 0.51, 25 °C) and the ionic
strength is capped at 0.5 M so as not to leave the validity range.

- **Where it lives**: `precipitation.py` → `calc_gamma_davies()`.
- **Why not to move it**: all published precipitation maps are computed this
  way. Changing the model (Pitzer, extended Debye-Hückel, another cap) forces
  recomputing the WHOLE precipitation sweep and revalidating against the paper.
  If it is ever done, it goes in as a new option, not as a replacement.

## 3. Batch PNG style — FROZEN (revisit if the paper asks for something else)

- **2D maps** (potential, field lines, ions, precipitation): **clean** version
  (no axes/title, colorbar/legend yes) — same as the console's "Save clean
  map?". `pc.guardar_figura(..., limpio=True)`.
- **1D profiles** and **total_ions** (2 panels): **full** figure with axes
  and titles (the clean version is not understandable on its own).
- **dpi = 300, bbox_inches="tight"** fixed in `pc.guardar_figura()`.
- Possible future flag `--annotated` / `--clean` in the batch if a per-run
  choice is needed; not added today so as not to bloat the CLI.

## 4. Legacy file names — FROZEN (unify in v2 if it becomes annoying)

Historical convention kept so as not to break existing scripts:

- maps: `{stem}_<suffix>_{V}V.png` (with the solution stem up front)
- profiles: `<analysis>_{V}V.png/.txt/.xlsx` (no stem; e.g. `ion_profile_`, `potential_profile_`)

Since everything lands in the solution folder, the stem is redundant; if it is
ever unified, do it in the `guardar()` of all modules at once.

## 5. Console: the precipitation TextBox now starts at 2.0

It used to start at 1.5 (historical value); it was aligned to `FACTOR_PAPER` so
that console, GUI and batch show the same by default. If 1.5 is missed, it is
one line in `procesar()` of `precipitation.py`.
