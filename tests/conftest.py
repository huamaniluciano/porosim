# -*- coding: utf-8 -*-
"""
Shared configuration for the POROSIM test suite.

The three pillars live in folders that START WITH A DIGIT (1_mesher,
2_solver, 3_extractor), so they are not importable Python packages via a plain
`import`. Here we add each folder to sys.path so their modules can be imported
by file name -- the same thing the launchers and the GUI do -- and we expose
useful paths and shared solver runs as fixtures.

Note on collisions: some module names (e.g. gui_app.py) are repeated across
pillars. The suite only imports uniquely named modules (capa1_modelo,
constantes, motor_pnp, porosim_comun, capa4_malla), so sharing sys.path is
safe. If a future test needs a repeated-name module, its import will have to
be isolated.

(POROSIM's own identifiers -- module names, functions, JSON keys -- are in
Spanish and kept verbatim: they are the actual API.)
"""
import sys
import pathlib

import pytest

# --- Import-order safeguard (legacy FEniCS quirk) ---------------------------
# In this environment `import h5py` only succeeds if it happens BEFORE dolfin
# (FEniCS): dolfin loads its own HDF5, and a later h5py import raises
# "ValueError: Not a datatype". The whole suite runs in a single process and the
# solver smoke test imports FEniCS, so we import h5py here first -- conftest is
# loaded before any test -- to lock in the working order for the mesher's XDMF
# export (meshio -> h5py). Guarded so a machine without h5py still collects.
try:
    import h5py  # noqa: F401  -- imported for its side effect: fixing import order
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The three pillar folders + the extractor's shared-modules layer.
_PILLAR_DIRS = [
    ROOT / "1_mesher",
    ROOT / "2_solver",
    ROOT / "3_extractor",
    ROOT / "3_extractor" / "modulos",
]
for _d in _PILLAR_DIRS:
    _p = str(_d)
    if _d.is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root():
    """Root of the POROSIM repository."""
    return ROOT


@pytest.fixture(scope="session")
def sales_json_path():
    """Path to the salt catalog (the solver's single source of truth)."""
    return ROOT / "2_solver" / "sales.json"


# --- Shared solver runs (used by T3 regression AND T4 physics validation) -----
# The PNP solver is the slow part; each distinct run is a session fixture so it
# executes once and its output is shared. Two cases are needed:
#   * film-less (mesh_demo-conical): regression (T3), conservation + neutrality.
#   * with a tip film (mesh_demo-tipfilm): analytical-Donnan check (T4).
_MESH_LESS = "mesh_demo-conical"
_PARAMS_LESS = "golden_solver_params.json"
_STEM_LESS = "demo-conical_KCl_100.0mM"                 # <m_name>_<salt>_<c0>

_MESH_FILM = "mesh_demo-tipfilm"
_PARAMS_FILM = "donnan_film_params.json"
_STEM_FILM = "demo-tipfilm_KCl_100.0mM"

_MESH_CYL = "mesh_demo-cylinder"
_PARAMS_CYL = "ohmic_cylinder_params.json"
_STEM_CYL = "demo-cylinder_KCl_100.0mM"


def _fenics_importable():
    """True if a fresh subprocess can import FEniCS (env actually usable)."""
    import subprocess
    return subprocess.run(
        [sys.executable, "-c", "import fenics"], capture_output=True, text=True
    ).returncode == 0


def _run_solver(repo_root, out_dir, mesh_dir, params):
    """Run the real `python solver.py run.json` entry point in a SUBPROCESS.

    Chosen over calling resolver() in-process because (a) it is what a reviewer
    runs, (b) FEniCS loads in a fresh process, avoiding any in-process
    HDF5/dolfin ordering issue, (c) PETSc/MUMPS global state stays isolated.
    Any equilibrium checkpoint the solver writes into RESULTS/equilibria/ (a
    fixed path with no override) is removed afterwards so the repo stays clean.
    Returns the completed subprocess.
    """
    import json
    import subprocess

    assert mesh_dir.is_dir(), f"committed test mesh missing: {mesh_dir}"
    cfg = {"mesh": str(mesh_dir), "output": str(out_dir), "params": params}
    cfg_path = out_dir / "run.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    eq_dir = repo_root / "RESULTS" / "equilibria"
    before = set(eq_dir.glob("*")) if eq_dir.is_dir() else set()
    solver = repo_root / "2_solver" / "solver.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(solver), str(cfg_path)],
            cwd=str(repo_root), capture_output=True, text=True, timeout=600,
        )
    finally:
        # Remove any checkpoint files this run created (keep the repo clean).
        if eq_dir.is_dir():
            for f in eq_dir.glob("*"):
                if f not in before:
                    f.unlink()

    assert proc.returncode == 0, \
        f"solver failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return proc


@pytest.fixture(scope="session")
def solver_run(repo_root, tmp_path_factory):
    """Run the PNP solver ONCE on the committed FILM-LESS demo mesh with the
    golden params; return {"dir", "iv", "h5", "sim_json"}. Shared by the
    regression test (T3) and the conservation/neutrality tests (T4). Skips if
    FEniCS is not usable in this environment.
    """
    import json

    if not _fenics_importable():
        pytest.skip("FEniCS not usable in this environment (see T1 _require notes)")

    data = repo_root / "tests" / "data"
    params = json.loads((data / _PARAMS_LESS).read_text(encoding="utf-8"))
    out_dir = tmp_path_factory.mktemp("solver_out")
    proc = _run_solver(repo_root, out_dir, data / _MESH_LESS, params)

    iv = out_dir / f"IV_curve_{_STEM_LESS}.txt"
    assert iv.is_file(), \
        f"solver did not produce the I-V curve.\nstdout tail:\n{proc.stdout[-1500:]}"
    return {
        "dir": out_dir,
        "iv": iv,
        "h5": out_dir / f"Solutions_{_STEM_LESS}.h5",
        "sim_json": out_dir / f"Solutions_{_STEM_LESS}_sim.json",
    }


@pytest.fixture(scope="session")
def solver_iv(solver_run):
    """Path to the I-V curve of the shared film-less run (see `solver_run`)."""
    return solver_run["iv"]


@pytest.fixture(scope="session")
def solver_run_film(repo_root, tmp_path_factory):
    """Run the PNP solver ONCE on the committed TIP-FILM demo mesh. The wall
    charge is zero, so the film interior is a pure Donnan phase -- ideal to check
    the numerical potential against the analytical Donnan value the solver
    exports. Returns {"dir", "h5", "sim_json"}. Skips if FEniCS is not usable.
    """
    import json

    if not _fenics_importable():
        pytest.skip("FEniCS not usable in this environment (see T1 _require notes)")

    data = repo_root / "tests" / "data"
    params = json.loads((data / _PARAMS_FILM).read_text(encoding="utf-8"))
    out_dir = tmp_path_factory.mktemp("solver_film_out")
    proc = _run_solver(repo_root, out_dir, data / _MESH_FILM, params)

    h5 = out_dir / f"Solutions_{_STEM_FILM}.h5"
    assert h5.is_file(), \
        f"solver did not produce the solution .h5.\nstdout tail:\n{proc.stdout[-1500:]}"
    return {
        "dir": out_dir,
        "h5": h5,
        "sim_json": out_dir / f"Solutions_{_STEM_FILM}_sim.json",
    }


@pytest.fixture(scope="session")
def solver_run_cylinder(repo_root, tmp_path_factory):
    """Run the PNP solver ONCE on the committed UNCHARGED cylinder mesh
    (sigma = 0), which is a purely ohmic resistor. Returns {"dir", "iv",
    "sim_json"}. Skips if FEniCS is not usable in this environment.
    """
    import json

    if not _fenics_importable():
        pytest.skip("FEniCS not usable in this environment (see T1 _require notes)")

    data = repo_root / "tests" / "data"
    params = json.loads((data / _PARAMS_CYL).read_text(encoding="utf-8"))
    out_dir = tmp_path_factory.mktemp("solver_cyl_out")
    proc = _run_solver(repo_root, out_dir, data / _MESH_CYL, params)

    iv = out_dir / f"IV_curve_{_STEM_CYL}.txt"
    assert iv.is_file(), \
        f"solver did not produce the I-V curve.\nstdout tail:\n{proc.stdout[-1500:]}"
    return {
        "dir": out_dir,
        "iv": iv,
        "sim_json": out_dir / f"Solutions_{_STEM_CYL}_sim.json",
    }
