# -*- coding: utf-8 -*-
"""
Mixtures of salts in the PNP solver (v1.1.0, N ions) -- end to end, FEniCS.

  * KCl 100 + HCl 10 mM (3 ions, 4 fields) against the binary salt it is
    equivalent to (110 mM, cation with D_eff; 3 fields): the I-V must coincide.
    K+ and H+ share z and their bulk ratio, so c_H/c_K is uniform in the pore
    and the mixture is exactly that binary salt (see conftest). A wrong sum in
    rho, in the current or in the boundary conditions of the extra ion breaks
    the equality.
  * The mixture's _sim.json carries the ions in the order of U and leaves
    c0_mM = null (a binary reader must fail instead of using a wrong c0).
  * The extractor reads the 4-field solution: the potential map is drawn and
    the ion maps are skipped (aplica() = False).

Marked `slow`.
"""
import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

np = pytest.importorskip("numpy")

_RTOL = 1e-6


def test_mixture_equals_the_equivalent_binary_salt(solver_run_mixture,
                                                   solver_run_effective_salt):
    mezcla = np.loadtxt(solver_run_mixture["iv"], skiprows=1)
    binaria = np.loadtxt(solver_run_effective_salt["iv"], skiprows=1)
    assert mezcla.shape == binaria.shape
    np.testing.assert_allclose(mezcla[:, 0], binaria[:, 0])
    np.testing.assert_allclose(mezcla[:, 1:], binaria[:, 1:], rtol=_RTOL, atol=1e-9,
                               err_msg="mixture and equivalent binary salt differ")


def test_mixture_sim_json(solver_run_mixture):
    sim = json.loads(solver_run_mixture["sim_json"].read_text(encoding="utf-8"))["simulation"]
    assert sim["c0_mM"] is None and sim["salt"] is None
    assert [(i["symbol"], i["c0_mM"]) for i in sim["ions"]] == \
        [("K⁺", 100.0), ("Cl⁻", 110.0), ("H⁺", 10.0)]
    assert sim["electrolyte_label"] == "KCl_100mM+HCl_10mM"
    assert sim["electrolyte"] == [{"salt": "KCl", "c0_mM": 100.0},
                                  {"salt": "HCl", "c0_mM": 10.0}]
    assert solver_run_mixture["h5"].is_file()


@pytest.mark.parametrize("module, dibuja", [("potential", True), ("ions", False)])
def test_extractor_on_a_mixture(solver_run_mixture, repo_root, module, dibuja):
    sol_dir, h5 = solver_run_mixture["dir"], solver_run_mixture["h5"]
    launcher = repo_root / "3_extractor" / "launch_extractor.py"
    before = set(sol_dir.glob("*"))
    proc = subprocess.run(
        [sys.executable, str(launcher), str(h5), module, "--voltage", "+0.10"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    pngs = [f for f in set(sol_dir.glob("*")) - before if f.suffix == ".png"]
    assert bool(pngs) is dibuja, f"{module}: {[f.name for f in pngs]}\n{proc.stdout[-800:]}"
