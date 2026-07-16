# POROSIM · Pillar 2 — PNP Solver

Solves the strongly coupled Poisson-Nernst-Planck system in 2D axisymmetric
form (FEniCS/dolfin 2019.1.0) on the Mesher's meshes (Pillar 1). **The physics
is exactly that of the solver that produced the paper results** — the refactor
into two modes only changed the packaging (verified by regression: I-V curve
identical digit by digit).

## Two usage modes

```bash
python solver.py                            # INTERACTIVE: console prompts
python launch_solver.py                     # INTERACTIVE: web GUI (Streamlit);
                                            #   without streamlit falls back to console
python solver.py params.json                # BATCH: no questions, iterable
python solver.py params.json --mesh PATH    # HYBRID: same physics, another mesh
python solver.py help                       # params.json field guide
```

The GUI (`gui_app.py` + `gui_dibujo_solver.py`) shows the domain with the
physics colored (blue = negative charge, red = positive, intensity =
magnitude), the proto-plot of the voltage sampling, and warns if a compatible
equilibrium checkpoint exists before launching. It produces the SAME `cfg` as
the console and the batch → same engine, same simulation. Note: the drawing
reuses `perfil_radio` from the mesher (requires `1_mesher/` in the repo).

All modes call the same `resolver(cfg)` of `motor_pnp.py`, so for the same
parameters the simulation is identical.

## File map

| Role | Files |
|---|---|
| **Core** (physics) | `motor_pnp.py` (`resolver(cfg)`, the paper's physics) · `constantes.py` · `sales.json` · `solver.py` (batch validation + console entry) |
| **Web GUI** | `gui_app.py` · `gui_dibujo_solver.py` (reuses `perfil_radio` from the mesher) |
| **Launcher** | `launch_solver.py` (no args → GUI; batch delegates to `solver.py`) |
| **Console equivalent to the GUI** | `console_backup/preguntas.py` (the prompts the GUI mimics; used by `python solver.py` with no arguments) |
| **Examples** | `examples/params_*.json` |


`--mesh` overrides the JSON `"mesh"` field: a single params.json defines the
physics and a shell loop sweeps the geometries, without editing files:

```bash
for m in ../1_mesher/meshes/*/; do
    python solver.py my_physics.json --mesh "$m"
done
```

(each run goes to its own default subfolder `<mesh>_<c0>mM`, so the sweep does
not overwrite itself).

## Inputs and outputs (contract)

**Reads** (from the mesher): `<name>_domain.xdmf` + `<name>_facets.xdmf` +
`<name>_limits.json`. The `_limits.json` is the source of truth about which
films the geometry has: the solver detects the films from it and only asks for
(or requires in the JSON) the charge of each one.

**Writes** (into `RESULTS/solutions/<subfolder>/`, or wherever `output`
says):

| File | Content |
|---|---|
| `Solutions_<mesh>_<c0>mM.h5` | mesh + solution U(φ, ln c₊, ln c₋) per voltage |
| `IV_curve_<mesh>_<c0>mM.txt` | table V / I_in / I_out [nA] |
| `Solutions_..._sim.json` | full metadata (geometry + simulation + salt) — read by the Extractor |

## Architecture

```
solver.py       entry: 2 modes + help + params.json validation
constantes.py   §1-3 of the original solver: universal physics, numerical
                knobs (Newton/MUMPS/tolerances), catalogs and defaults
preguntas.py    §4-5: the interactive flow (same texts) → produces cfg
motor_pnp.py    §6: the PHYSICS (untouched) — resolver(cfg): loads mesh,
                assembles PNP, 3 stages, exports
sales.json      salt catalog (valences, D, solubility, Ksp)
```

The 3 stages: **(1)** surface-charge ramp σ at V=0 (strict tolerances) →
**(1b)** volumetric-charge ramp ρ_film, sequential per film → **(2)** voltage
sweep ±V_max with adaptive step subdivision.

**Equilibrium checkpoint**: stages 1+1b are deterministic (~85 MUMPS
factorizations); the result is saved in `RESULTS/equilibria/` with the full
parameter key and reused if the equilibrium parameters match (batch:
`reuse_equilibrium: "auto"`).
*Physical note:* the key includes the valences $z_p/z_m$ but not the salt name
or its diffusivities ($D$), since at $V=0$ (thermodynamic equilibrium) the flux
is zero and the electrostatic distribution is identical for any salt of the
same type (e.g., reusing the KCl equilibrium for KClO4 or NaCl is physically
correct, as they are all symmetric 1:1 salts).


**Logarithmic variables**: u± = ln(c±/c0) is solved, which guarantees c± > 0.
The voltage enters non-dimensionalized (φ in units of RT/F).

## params.json guide

`python solver.py help` prints the table with the current defaults. Minimal
complete example (see `examples/`):

```json
{
  "mesh": "../1_mesher/meshes/conico_chico/conico_chico_domain.xdmf",
  "params": {
    "salt": "KCl",
    "c0_mM": 100.0,
    "sigma_e_nm2": -1.0,
    "sigma_on_rings": true,
    "V_max_V": 1.0,
    "n_steps": 11
  }
}
```

- σ is in **e/nm²** (1 e/nm² = 0.1602 C/m²; the negative sign = negative wall).
- If the mesh has films, `"films"` is required: one entry per film, with
  `"type"` from the catalog (`0.66M`/`1M`/`2M`/`4M`) or a free `"molar_charge"`.
  The order of the list defines the ramp order.
- Salt: a name from the `sales.json` catalog (e.g. `"KCl"`, `"KClO4"`, `"NaCl"`)
  or a custom object (dictionary) to simulate custom species.
  Example of a custom salt directly in the JSON:
  ```json
  "salt": {
    "name": "MySalt2_1",
    "cation": { "z": 2, "D_m2s": 0.79e-9 },
    "anion":  { "z": -1, "D_m2s": 2.03e-9 },
    "soluble": false,
    "Ksp_M2": 1.5e-3
  }
  ```
  *(Note: adding a permanent salt to the general catalog is as simple as adding
  an entry to `sales.json`, without touching a single line of code.)*


## Known limitations

1. **Concentration gradient: NOT implemented.** The structure is in place (the
   `_sim.json` already exports `c0_inlet_mM`/`c0_outlet_mM`) but the Δc ramp is
   missing. In batch, `c0_inlet_mM ≠ c0_outlet_mM` gives an explicit error; in
   interactive mode it warns and uses a symmetric concentration.
2. **Highly charged film + very dilute electrolyte** (contrast c_fix/c0 ≳ 50,
   e.g. a 0.75 M film with KCl 10 mM): the system becomes stiff. The symptom
   can be Newton failure **or a current that collapses non-physically**
   (converges to a bad solution without raising an error — the automatic
   subdivision does not catch it). **Validated recipe**: slower ramps — raise
   `n_steps` (voltage) and/or `n_steps_film` (ρ_film ramp). Quality check:
   verify conservation |I_in − I_out| ≪ |I_in| at each point of the curve.
3. The sweep cuts the branch (does not abort the run) if a point does not
   converge even by subdividing: the I-V curve stops at the last converged
   voltage.
