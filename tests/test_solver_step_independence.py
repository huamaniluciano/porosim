# -*- coding: utf-8 -*-
"""
Voltage-step independence of the I-V curve (Pillar 2, added in v1.1.0).

v1.0.0 swept the voltage with Newton tolerances (1e-12, 1e-8) that stopped
Newton after a single linearised iteration, so the I-V curve depended on the
voltage step (up to 22 % at +-0.1 V with 0.1 V steps). A converged steady state
cannot depend on how it was reached: the same case solved with 0.1 V and with
0.05 V steps must give the same currents at the common voltages.

Reuses the shared golden run (`solver_run`, n_steps = 3) and one extra run with
half the step (`solver_run_fine`, n_steps = 5). Marked `slow`.
"""
import json

import pytest

pytestmark = pytest.mark.slow

# With the strict tolerance the two step sizes agree to < 0.005 % on the paper
# geometry (ARREGLOS PARA POROSIM/SOLVER_VOLTAGE_TOLERANCE.md); with the old
# tolerance they differed by several percent. 0.05 % separates both regimes
# with a wide margin on either side.
_RTOL = 5e-4


def _iv_por_voltaje(path):
    np = pytest.importorskip("numpy")
    data = np.loadtxt(path, skiprows=1)
    return {round(float(v), 4): (float(i_in), float(i_out)) for v, i_in, i_out in data}


def test_iv_does_not_depend_on_voltage_step(solver_run, solver_run_fine):
    """I(V) at +-0.1 V and +-0.2 V matches between 0.1 V and 0.05 V steps."""
    grueso = _iv_por_voltaje(solver_run["iv"])
    fino = _iv_por_voltaje(solver_run_fine["iv"])
    comunes = sorted(v for v in grueso if v in fino and abs(v) > 1e-9)
    assert comunes == [-0.2, -0.1, 0.1, 0.2], f"unexpected common voltages: {comunes}"
    for v in comunes:
        for k, nombre in enumerate(("I_in", "I_out")):
            a, b = grueso[v][k], fino[v][k]
            rel = abs(a - b) / max(abs(a), abs(b))
            assert rel < _RTOL, (
                f"{nombre} at V={v:+.2f} V depends on the voltage step: "
                f"{a:.6f} nA (0.1 V steps) vs {b:.6f} nA (0.05 V steps), "
                f"rel. diff {rel:.2e}. Newton is probably stopping before "
                f"convergence (see TOL_VOLTAJE in 2_solver/constantes.py)."
            )


def test_no_shallow_newton_steps(solver_run, solver_run_fine):
    """No voltage step converges in fewer than 2 Newton iterations, and the
    run records the voltage-sweep tolerance it used."""
    for run in (solver_run, solver_run_fine):
        sim = json.loads(run["sim_json"].read_text(encoding="utf-8"))
        codigos = [w.get("code") for w in sim.get("warnings", [])]
        assert "newton_few_iterations" not in codigos, sim["warnings"]
        assert sim["simulation"]["tol_voltage_abs"] <= 1e-20
