# -*- coding: utf-8 -*-
"""
T4 - Physical validation tests (Pillar 2).

Unlike the regression test (T3), these assert against PHYSICS -- laws and
analytical formulas known a priori -- not against a stored golden number. The
"expected" value IS the physics; the tolerance is the numerical-quality budget.
They reuse the single solver run from the shared `solver_iv` fixture
(conftest.py), so no extra solve is needed.

These numbers are also the seed of the "Validation" table of the companion
chemistry paper. Marked `slow`.
"""
import pytest

pytestmark = pytest.mark.slow


# The solver flags an inlet/outlet current mismatch as a soft warning at 5%
# (motor_pnp.py, sanity checks). Here we assert a stricter 1% budget on the
# well-resolved demo case (observed max mismatch ~0.12%). A voltage point whose
# mean current is essentially numerical zero (V=0) carries no meaningful ratio
# and is skipped.
_FLOOR_nA = 1e-6
_MAX_REL_MISMATCH = 1e-2


def test_current_conservation(solver_iv):
    """Charge conservation: at steady state the physical inlet and outlet
    currents must be equal (there is no current-limiting physics inside the PNP
    motor). The I-V file columns are already the PHYSICAL currents (inlet =
    -I_in_raw, outlet = I_out_raw; see motor_pnp.py line ~673), so conservation
    means column 2 ~= column 3 at every non-trivial voltage.

    The 'expected' here is not a stored number but the law itself: the relative
    imbalance must vanish, up to the mesh's numerical-quality budget.
    """
    np = pytest.importorskip("numpy")
    data = np.loadtxt(solver_iv, skiprows=1)   # columns: V, I_in(phys), I_out(phys)

    checked = 0
    for v, i_in, i_out in data:
        mean_mag = (abs(i_in) + abs(i_out)) / 2.0
        if mean_mag <= _FLOOR_nA:
            continue                            # V=0: numerical zero, ratio meaningless
        rel = abs(i_in - i_out) / mean_mag
        assert rel < _MAX_REL_MISMATCH, (
            f"current not conserved at V={v:+.2f} V: "
            f"|I_in - I_out|/mean = {rel * 100:.2f}% "
            f"(budget {_MAX_REL_MISMATCH * 100:.0f}%)"
        )
        checked += 1

    assert checked > 0, "no voltage point had a non-zero current to check"


def test_bulk_electroneutrality(solver_run):
    """At V=0, far from the charged pore wall and from the reservoir edges, the
    fluid must be bulk: electrically neutral (z+ c+ + z- c- ~= 0) and at the
    imposed bulk concentration (c+ ~= c- ~= c0). Uses the extractor's own loader
    (porosim_comun) to read the P1 nodal concentration fields (c = c0*exp(u))
    from the solution .h5.

    The 'expected' is the physics, not a stored number: bulk neutrality and bulk
    concentration. Within a few Debye lengths of the charged wall the double
    layer breaks local neutrality (c+ != c-); that region is deliberately
    excluded by sampling the inlet-reservoir interior.
    """
    np = pytest.importorskip("numpy")
    try:
        import porosim_comun as pc          # on sys.path via conftest
    except Exception as exc:                 # noqa: BLE001 -- FEniCS/env issues
        pytest.skip(f"extractor loader unavailable: {exc!r}")

    h5 = solver_run["h5"]
    sol = pc.cargar_solucion(h5)
    meta, mesh, V = sol["meta"], sol["mesh"], sol["V"]
    sim = meta.get("simulation", {})
    sal = pc.info_sal(sim)
    c0 = float(sim["c0_mM"])
    z_p, z_m = sal["z_p"], sal["z_m"]

    assert "+0.00" in sol["labels"], f"no V=0 solution stored; labels={sol['labels']}"
    fields = pc.leer_campos(h5, mesh, V, "+0.00")
    dm = pc.datos_malla(mesh)
    z, r = dm["z_nm"], dm["r_nm"]                     # vertex coordinates [nm]
    c_p = c0 * np.exp(fields["up"])
    c_m = c0 * np.exp(fields["um"])

    # Bulk band: inlet-reservoir interior, away from the membrane/pore (small z)
    # and from the outer reservoir wall (moderate r).
    z_tip = meta["z_tip"] * 1e9
    R_res = meta["R_res"] * 1e9
    bulk = (z > 0.10 * z_tip) & (z < 0.30 * z_tip) & (r < 0.60 * R_res)
    assert bulk.sum() >= 3, f"too few bulk nodes to sample ({int(bulk.sum())})"

    # (1) Local electroneutrality in the bulk.
    imbalance = np.abs(z_p * c_p[bulk] + z_m * c_m[bulk]) / c0
    assert imbalance.max() < 1e-2, \
        f"bulk not electroneutral: max |z+c+ + z-c-|/c0 = {imbalance.max():.2e}"

    # (2) Bulk concentration equals the imposed c0 (within 1%).
    for name, c in (("c+", c_p[bulk]), ("c-", c_m[bulk])):
        dev = np.abs(c - c0).max() / c0
        assert dev < 1e-2, f"bulk {name} deviates from c0 by {dev * 100:.2f}%"


def test_donnan_potential(solver_run_film):
    """Donnan equilibrium: a charged film in contact with a bulk electrolyte
    develops a potential plateau in its interior given analytically (1:1) by
    phi_D = (RT/F)*asinh(c_fix / (2*c0)). The solver computes this analytical
    value and exports it as `phi_D_anal_mV`; here we check the NUMERICAL
    equilibrium potential inside the film matches it.

    The 'expected' is the analytical formula, not a stored number. The film has
    finite thickness, so its potential is not uniform (it drops to bulk at the
    film/water interface); the Donnan PLATEAU is the interior value, estimated
    robustly by a high percentile of |phi| over the film region. With zero wall
    charge the film is the only charge, so the interior is a pure Donnan phase.
    """
    np = pytest.importorskip("numpy")
    try:
        import porosim_comun as pc          # on sys.path via conftest
    except Exception as exc:                 # noqa: BLE001 -- FEniCS/env issues
        pytest.skip(f"extractor loader unavailable: {exc!r}")

    h5 = solver_run_film["h5"]
    sol = pc.cargar_solucion(h5)
    meta, mesh, V = sol["meta"], sol["mesh"], sol["V"]
    sim = meta.get("simulation", {})

    films = pc.films_activos_de(meta)
    assert films, "no active film found in the solution metadata"
    film = films[0]
    phi_D_anal_mV = film["phi_D_mV"]
    assert phi_D_anal_mV is not None, "solver did not export phi_D_anal_mV"

    assert "+0.00" in sol["labels"], f"no equilibrium (V=0) solution; labels={sol['labels']}"
    V_T = pc.voltaje_termico(float(sim.get("T_K", 298.15)))     # thermal voltage [V]
    fields = pc.leer_campos(h5, mesh, V, "+0.00")
    dm = pc.datos_malla(mesh)
    z, r = dm["z_nm"], dm["r_nm"]
    phi_mV = fields["phi_adim"] * V_T * 1e3                     # dimensionless -> mV

    # Film region: its z-slab (film/water interface -> membrane mouth) within the
    # pore-mouth radius. The Donnan plateau is the deep-interior value.
    z_int = film["z_int"] * 1e9
    z_memb = film["z_memb"] * 1e9
    R_tip = meta["R_tip"] * 1e9
    in_film = (z >= z_int) & (z <= z_memb) & (r <= R_tip)
    assert in_film.sum() >= 20, f"too few film nodes to sample ({int(in_film.sum())})"

    # High percentile = the interior plateau (robust to the interface transition
    # and to single outlier nodes).
    phi_plateau_mV = np.percentile(phi_mV[in_film], 95)
    rel = abs(phi_plateau_mV - phi_D_anal_mV) / abs(phi_D_anal_mV)
    assert rel < 0.05, (
        f"numerical Donnan {phi_plateau_mV:+.2f} mV vs analytical "
        f"{phi_D_anal_mV:+.2f} mV: {rel * 100:.1f}% (budget 5%)"
    )


def test_ohmic_conductance_uncharged_cylinder(solver_run_cylinder):
    """An uncharged cylindrical pore is a pure ohmic resistor. Its conductance
    G = I/V must match the analytical series resistance of the pore PLUS the Hall
    access resistance of its two openings:

        kappa    = (F^2/RT) * sum_i z_i^2 D_i c_i        (bulk conductivity)
        R_pore   = L / (kappa * pi * a^2)
        R_access = 2 * 1/(4 kappa a) = 1/(2 kappa a)     (both openings, Hall)
        G_theory = 1 / (R_pore + R_access)

    The 'expected' is the analytical formula, not a stored number. The access
    term matters: without it the numerical G is ~7% low (measured); with it the
    agreement is ~0.1%. (Note: the classic per-opening value is 1/(4 kappa a);
    the two openings in series give 1/(2 kappa a) total.)
    """
    np = pytest.importorskip("numpy")
    import json

    F, R_gas = 96485.3, 8.314

    sim = json.loads(solver_run_cylinder["sim_json"].read_text(encoding="utf-8"))
    S = sim["simulation"]
    T = float(S.get("T_K", 298.15))
    c0 = float(S["c0_mM"])                          # mM == mol/m^3
    salt = S["salt"]
    z_p, D_p = salt["cation"]["z"], salt["cation"]["D_m2s"]
    z_m, D_m = salt["anion"]["z"], salt["anion"]["D_m2s"]

    a = float(sim["R_tip"])                          # pore radius [m]
    assert abs(sim["R_tip"] - sim["R_base"]) <= 1e-12 * a, "mesh is not a cylinder"
    L = float(sim["L_pore"])                         # pore length [m]

    kappa = (F ** 2 / (R_gas * T)) * (z_p ** 2 * D_p + z_m ** 2 * D_m) * c0   # S/m
    R_pore = L / (kappa * np.pi * a ** 2)
    R_access = 1.0 / (2.0 * kappa * a)               # two Hall openings in series
    G_theory = 1.0 / (R_pore + R_access)

    iv = np.loadtxt(solver_run_cylinder["iv"], skiprows=1)   # V, I_in, I_out (nA)
    V, I_out_nA = iv[:, 0], iv[:, 2]
    nz = np.abs(V) > 1e-6
    assert nz.sum() >= 2, "need non-zero voltages to measure conductance"
    G_num = float(np.mean(I_out_nA[nz] * 1e-9 / V[nz]))       # S (I in A)

    rel = abs(G_num / G_theory - 1.0)
    assert rel < 0.05, (
        f"ohmic conductance {G_num:.3e} S vs theory {G_theory:.3e} S "
        f"(pore + Hall access): {rel * 100:.1f}% (budget 5%)"
    )

