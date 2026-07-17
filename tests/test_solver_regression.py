# -*- coding: utf-8 -*-
"""
T3 - Solver regression test (Pillar 2).

Compares the solver's I-V curve on the committed demo case against a golden
reference in tests/data/. The solver itself runs once in the shared `solver_iv`
fixture (see conftest.py), reused by the T4 physics-validation tests. This
formalizes the "digit-by-digit" regression check done by hand during refactors:
if a formula in motor_pnp.py changes, the currents shift and this test fails.

The run is deterministic on a given machine (direct MUMPS solver, no
randomness). The tolerance (rtol=1e-4) absorbs minor cross-platform numeric
noise yet is far tighter than any real formula change (which shifts currents by
>> 0.01%). Loosen it, WITH a note, only if CI shows genuine platform drift.

Marked `slow`: exclude during fast iteration with `-m "not slow"`.
"""
import pytest

pytestmark = pytest.mark.slow

_GOLDEN = "iv_golden_demo-conical_KCl_100mM.txt"


def test_iv_curve_shape(solver_iv):
    """The I-V table has the expected header and the symmetric voltage grid."""
    np = pytest.importorskip("numpy")
    header = solver_iv.read_text(encoding="utf-8").splitlines()[0]
    assert header.split() == ["Voltage(V)", "I_in(nA)", "I_out(nA)"], \
        f"unexpected header: {header!r}"
    data = np.loadtxt(solver_iv, skiprows=1)
    assert data.shape == (5, 3), f"expected 5 voltages x 3 columns, got {data.shape}"
    assert sorted(round(v, 3) for v in data[:, 0]) == [-0.2, -0.1, 0.0, 0.1, 0.2]


def test_iv_matches_golden(solver_iv, repo_root):
    """The freshly computed I-V curve matches the committed golden reference."""
    np = pytest.importorskip("numpy")
    golden = np.loadtxt(repo_root / "tests" / "data" / _GOLDEN, skiprows=1)
    fresh = np.loadtxt(solver_iv, skiprows=1)
    assert fresh.shape == golden.shape, f"shape drift: {fresh.shape} vs {golden.shape}"

    # Voltages are inputs: must match exactly. Currents: within tolerance.
    np.testing.assert_allclose(fresh[:, 0], golden[:, 0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        fresh[:, 1:], golden[:, 1:], rtol=1e-4, atol=5e-6,
        err_msg="I-V drifted from the golden reference beyond tolerance -- a "
                "formula in motor_pnp.py may have changed. (If this is only "
                "cross-platform numeric noise, loosen rtol here WITH a note.)",
    )
