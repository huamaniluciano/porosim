# -*- coding: utf-8 -*-
"""
Adaptive subdivision of the voltage step (Pillar 2, fixed in v1.1.0).

When Newton diverges on a voltage step, the solver goes back to the last
converged state and tries half the step. Up to v1.1.0 the retries reused the
same Newton solver, whose LU preconditioner had failed with NaN (PETSc
DIVERGED_PC_FAILED), so every retry failed at iteration 0, even 1/64 of the
step, and the branch stopped at that voltage. With the v1.0.0 tolerance a step
almost never diverged (Newton stopped after one linear step), so the bug showed
with the strict v1.1.0 tolerance, with large steps and with dilute
electrolytes (1 mM).

On the demo mesh a single 0 -> +-1 V step diverges and has to be subdivided.
The test asks for exactly that single step and checks that both ends of the
sweep are reached with the currents of a converged sweep in 0.05 V steps.
Marked `slow`.
"""
import json

import pytest

pytestmark = pytest.mark.slow

# Converged I(+-1 V) of this case, swept in 0.05 V steps (n_steps = 21) with the
# v1.1.0 tolerance: (V, I_in, I_out) in nA, as written in the I-V table.
_REFERENCIA = {-1.0: (-4.202930, -4.232222), +1.0: (5.656510, 5.622002)}
# Converged steady states agree to ~1e-6 whatever the path; 1e-4 leaves margin.
_RTOL = 1e-4


def test_one_volt_step_is_subdivided_and_converges(repo_root, tmp_path):
    np = pytest.importorskip("numpy")
    import conftest

    if not conftest._fenics_importable():
        pytest.skip("FEniCS not usable in this environment")
    data = repo_root / "tests" / "data"
    params = json.loads((data / conftest._PARAMS_LESS).read_text(encoding="utf-8"))
    params.update({"V_max_V": 1.0, "n_steps": 2})       # one 1 V step per branch
    proc = conftest._run_solver(repo_root, tmp_path, data / conftest._MESH_LESS, params)

    assert "[FAILED]" not in proc.stdout, proc.stdout[-2500:]
    iv = np.loadtxt(tmp_path / f"IV_curve_{conftest._STEM_LESS}.txt", skiprows=1)
    filas = {round(float(v), 4): (float(a), float(b)) for v, a, b in iv}
    assert sorted(filas) == [-1.0, 0.0, 1.0], \
        f"the sweep did not reach both ends: {sorted(filas)}\n{proc.stdout[-2500:]}"
    for v, (ref_in, ref_out) in _REFERENCIA.items():
        np.testing.assert_allclose(filas[v], (ref_in, ref_out), rtol=_RTOL,
                                   err_msg=f"I({v:+.0f} V) after subdividing the step")
