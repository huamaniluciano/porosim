# -*- coding: utf-8 -*-
"""
Regenerate the solver's golden I-V curve
(tests/data/iv_golden_demo-conical_KCl_100mM.txt), used by the regression test
tests/test_solver_regression.py.

Run it ONLY when the physics or the numerics change on purpose (v1.1.0: the
voltage-sweep Newton tolerance, see 2_solver/constantes.py), on a machine with
FEniCS, from the repository root:

    python tests/regenerate_golden.py            # compare, then write the new golden
    python tests/regenerate_golden.py --check    # compare only, write nothing

It solves the same case as the regression test (the committed film-less demo
mesh with tests/data/golden_solver_params.json) through the real solver entry
point, in a temporary folder, with the same helper as the test fixtures
(conftest._run_solver, which also removes the equilibrium checkpoint it
creates). Commit the new golden together with the change that motivated it.
"""
import pathlib
import shutil
import sys
import tempfile

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import json  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402  (same solver run as the `solver_run` fixture)

GOLDEN = AQUI / "data" / "iv_golden_demo-conical_KCl_100mM.txt"


def main(solo_comparar=False):
    if not conftest._fenics_importable():
        sys.exit("FEniCS is not importable in this Python: activate the porosim env.")
    data = AQUI / "data"
    params = json.loads((data / conftest._PARAMS_LESS).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="porosim_golden_") as tmp:
        out = pathlib.Path(tmp)
        print(">>> Running the solver on the demo mesh (a few seconds) ...")
        conftest._run_solver(conftest.ROOT, out, data / conftest._MESH_LESS, params)
        nueva = out / f"IV_curve_{conftest._STEM_LESS}.txt"
        actual = np.loadtxt(GOLDEN, skiprows=1)
        fresca = np.loadtxt(nueva, skiprows=1)
        print(f"\n{'V':>6} {'I_in old':>12} {'I_in new':>12} {'rel. diff':>10}")
        for (v, i_old, _), (_, i_new, _) in zip(actual, fresca):
            rel = abs(i_new - i_old) / max(abs(i_old), 1e-12) if abs(i_old) > 1e-9 else 0.0
            print(f"{v:+6.2f} {i_old:12.6f} {i_new:12.6f} {rel:10.2%}")
        if solo_comparar:
            print("\n--check: nothing written.")
            return
        shutil.copy(nueva, GOLDEN)
        print(f"\n✓ Golden updated: {GOLDEN.relative_to(conftest.ROOT)}")


if __name__ == "__main__":
    main(solo_comparar="--check" in sys.argv[1:])
