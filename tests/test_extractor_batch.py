# -*- coding: utf-8 -*-
"""
T5 - Extractor batch test (Pillar 3).

Runs the extractor's real batch entry point

    python launch_extractor.py <solution.h5> <module> --voltage +0.00 --save-data

on the shared demo solution and checks it produces the expected figures (PNG)
and, for a module that exports tabular data, the numerical data file too.

The extractor runs in a SUBPROCESS: it is what a reviewer runs, and FEniCS loads
in a fresh process. It reuses the `solver_run` solution (conftest) so no extra
solve is needed -- the extractor reads the mesh and fields straight from the
solver's Solutions_*.h5, closing the mesher -> solver -> extractor loop.

Marked `slow`.
"""
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

# (module name, whether --save-data should also emit a .txt/.xlsx data file)
_MODULES = [
    ("potential", False),                 # 2D potential map -> PNG
    ("ions", False),                      # 2D ion maps -> one PNG per ion
    ("axis_profile_potential", True),     # axial profile -> PNG + .txt data
]


@pytest.mark.parametrize("module, exports_data", _MODULES)
def test_extractor_batch_produces_outputs(solver_run, repo_root, module, exports_data):
    """Each representative module writes at least one non-empty PNG into the
    solution folder; the profile module also writes a numerical data file."""
    sol_dir = solver_run["dir"]
    h5 = solver_run["h5"]
    launcher = repo_root / "3_extractor" / "launch_extractor.py"

    before = set(sol_dir.glob("*"))
    proc = subprocess.run(
        [sys.executable, str(launcher), str(h5), module,
         "--voltage", "+0.00", "--save-data"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"extractor '{module}' failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )

    new = set(sol_dir.glob("*")) - before
    pngs = [f for f in new if f.suffix == ".png"]
    assert pngs, f"module '{module}' produced no PNG.\nstdout:\n{proc.stdout[-1200:]}"
    assert all(f.stat().st_size > 0 for f in pngs), f"module '{module}' produced an empty PNG"

    if exports_data:
        data = [f for f in new if f.suffix in (".txt", ".xlsx")]
        assert data, f"module '{module}' with --save-data produced no data file"
        assert all(f.stat().st_size > 0 for f in data), \
            f"module '{module}' produced an empty data file"
        # The .txt export must parse as a numeric table (header lines start '#').
        np = pytest.importorskip("numpy")
        txts = [f for f in data if f.suffix == ".txt"]
        assert txts, f"module '{module}' produced no .txt data file"
        arr = np.loadtxt(txts[0])
        assert arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 2, \
            f"data file {txts[0].name} does not parse as a table (shape {arr.shape})"


def test_precipitation_skipped_on_soluble_salt(solver_run, repo_root):
    """A module whose aplica(meta) is False for this solution -- precipitation on
    a fully soluble salt (KCl) -- is skipped cleanly: the batch exits 0 and
    writes no figure, instead of crashing."""
    sol_dir = solver_run["dir"]
    h5 = solver_run["h5"]
    launcher = repo_root / "3_extractor" / "launch_extractor.py"

    before = set(sol_dir.glob("*"))
    proc = subprocess.run(
        [sys.executable, str(launcher), str(h5), "precipitation", "--voltage", "+0.00"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"a skipped module should exit 0, got {proc.returncode}:\n{proc.stderr[-1000:]}"
    )
    new_pngs = [f for f in set(sol_dir.glob("*")) - before if f.suffix == ".png"]
    assert not new_pngs, \
        f"precipitation should produce no PNG on a soluble salt, got: {[f.name for f in new_pngs]}"
