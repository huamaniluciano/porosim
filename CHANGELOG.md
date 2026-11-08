# Changelog

All notable changes to POROSIM. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.1.0] - Unreleased

Fixes and features gathered from the copies of the code used for the
precipitation, pH and memristor papers. Runs written by v1.0.0 (`.h5`,
`_sim.json`, I-V tables) are still read by the extractor.

### Fixed

- **Solver: the I-V curve depended on the voltage step.** The Newton
  tolerance of the voltage sweep, `(1e-12, 1e-8)`, stopped after one
  linearised iteration, so currents changed with the step (up to 22 % at
  ±0.1 V with 0.1 V steps, about 1.4 % with 0.05 V steps). It is now
  `(1e-20, 1e-30)`, like the equilibrium ramps (3 to 4 iterations per step,
  about 1.75× the wall time). New run parameters `tol_voltage_abs` and
  `tol_voltage_rel` reproduce v1.0.0 runs, a `newton_few_iterations` warning
  flags steps that converge in fewer than 2 iterations, and a new test checks
  that two voltage steps give the same currents. **Curves computed with v1.0.0
  should be recomputed.** The golden I-V of the regression test was
  regenerated (`python tests/regenerate_golden.py`): its currents move by 0.4
  to 8.5 %, and the old ±0.1 V points, exactly symmetric, were the linear
  response of a single Newton step. With the v1.0.0 tolerances the v1.1.0
  solver gives the old golden to the last printed digit.
- **Solver: asymmetric salts had a charged bulk.** A 2:1 salt such as CaCl₂
  got c0 on both ions; it now runs with its neutral stoichiometry
  (Ca²⁺ c0, Cl⁻ 2·c0).
- **Extractor, potential maps and field lines** (from the memristor paper
  copy): E_r = 0 on the symmetry axis (the interpolator gave a spurious radial
  field there); zoomed views compute the field on the zoom window (they used to
  have about one grid node, so no field lines), with the interpolator built on
  the window only (~90× faster on a 650k-node mesh);
  zoomed field lines are exactly symmetric; no colour jump at the end of the
  colorbar; colorbar ticks never fall outside the bar; axes end at the data;
  clean exports keep the frame and the tick marks.
- **Extractor: the cross-section average crashed with NumPy ≥ 2.4**
  (`np.trapz` was removed).
- **Demo film mesh** (`RESULTS/meshes/demo-conical_tipfilm`) could not be
  loaded: its `.xdmf` pointed to another folder's `.h5`.
- **Solver examples** (`2_solver/examples/params_*.json`) pointed to paths from
  before the folder renames.
- `environment.yml` uses conda-forge only.

### Added

- **Mixtures of salts (N ions)** in the solver: an `"electrolyte"` block such as
  `[{"salt": "KCl", "c0_mM": 100}, {"salt": "HCl", "c0_mM": 10}]`, one
  Nernst-Planck equation per ion, ions shared by two salts are merged, an
  electroneutral bulk is checked, and the Donnan potential of a film is solved
  for N ions. With a single symmetric salt the equations are exactly those of
  v1.0.0 (same file names and `_sim.json` fields, plus an `"ions"` list). HCl
  and KOH added to `sales.json`. The extractor draws the potential, the I-V and
  its metrics for a mixture.
- **Precipitation analyses** (sparingly soluble salts): the supersaturation
  threshold S is a parameter (default 2, as in v1.0.0); ternary map
  (undersaturated / supersaturated below S / precipitates) and continuous
  log₁₀(Q_act/Ksp) map; new global module `precipitation_onset` (max Q_act/Ksp
  vs V and the onset voltage of each polarity), also in the GUI.
- **I-V metrics** (new global module `iv_metrics`): rectification factor,
  conductance G0, exponent n = d ln I / d ln V on 0.7–1 V and S-index of each
  branch, local slope; shown in the GUI and in the solution summary.
- **Batch options**: `--opt key=value` passes options to a module
  (`python launch_extractor.py list` shows them) and `--format pdf|svg` saves
  the figures in a vector format.
- **Editable text in vector exports**: PDF with embedded TrueType fonts and SVG
  with text elements, in batch and in the GUI downloads.
- **Fixed potential colour scale** (opt-in, `--opt phi_ref=1.2` or the GUI
  box) to compare maps of different voltages or runs, with a warning when the
  data exceed the scale.
- **Validation against COMSOL** in the README (currents within ~1 %).
- Tests: 31 → 105 (electrolyte logic, mixtures, voltage-step independence,
  extractor field and export helpers, precipitation, I-V metrics, the GUI
  driven with Streamlit's AppTest without FEniCS).

## [1.0.0] - 2026-07-16

First public release: mesher, PNP solver and extractor for 2D axisymmetric
nanopores, with web GUIs, batch modes and a test suite.

[1.1.0]: https://github.com/huamaniluciano/porosim/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/huamaniluciano/porosim/releases/tag/v1.0.0
