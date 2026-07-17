# -*- coding: utf-8 -*-
"""
T1 - Smoke tests: importing the three pillars.

Goal: catch gross breakage (syntax errors, broken imports, missing
dependencies) in seconds by importing each pillar's core module and checking
that it exposes its main API. These run no physics and build no mesh: they
only confirm that "the engine turns over".

Modules that need FEniCS or gmsh are guarded so the suite still runs in a
minimal environment (e.g. without dolfin), skipping them instead of failing.
In CI (full environment) they do run.

POROSIM's own module/function names are in Spanish (capa1_modelo, motor_pnp,
mallar, resolver, ...) and are referenced verbatim: they are the real API.
"""
import importlib

import pytest


def _require(modname):
    """Import an optional heavy dependency; if it can't be imported -- absent
    OR with a broken environment (e.g. FEniCS with a mis-pointed
    pkg-config/compiler) -- SKIP the test with the reason, instead of failing.
    Fixing a broken FEniCS/gmsh environment is not a smoke test's job: the
    functional tests (T3/T4) exercise the solver for real and would fail
    loudly. Unlike pytest.importorskip, we catch ANY exception here, not only
    ImportError.
    """
    try:
        return importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001 -- deliberate: ImportError, RuntimeError, ...
        pytest.skip(f"'{modname}' unavailable in this environment: {exc!r}")


# -- Pillar 1: MESHER ---------------------------------------------------------
def test_mesher_capa1_modelo_imports():
    """capa1_modelo: declarative topology, no heavy dependencies."""
    import capa1_modelo as m
    assert hasattr(m, "Params")
    assert isinstance(m.TAGS, dict) and m.TAGS, "TAGS must be a non-empty dict"
    assert callable(m.construir_estaciones)


def test_mesher_capa4_malla_imports():
    """capa4_malla: Gmsh mesher (public entry point `mallar`). Needs gmsh."""
    _require("gmsh")
    _require("meshio")
    import capa4_malla as m
    assert callable(m.mallar)


# -- Pillar 2: SOLVER ---------------------------------------------------------
def test_solver_constantes_imports():
    """constantes: universal physics + numeric knobs. Importable without FEniCS."""
    import constantes as c
    assert c.F_CONST == pytest.approx(96485.3)
    assert c.R_GAS == pytest.approx(8.314)
    assert c.EPS_0 > 0
    assert c.NEWTON_MAX_ITER >= 1


def test_solver_motor_pnp_imports():
    """motor_pnp: PNP engine (public entry point `resolver`). Needs FEniCS."""
    _require("fenics")
    import motor_pnp as m
    assert callable(m.resolver)


# -- Pillar 3: EXTRACTOR ------------------------------------------------------
def test_extractor_porosim_comun_imports():
    """porosim_comun: the extractor's shared layer (lightweight import)."""
    import porosim_comun as pc
    for fn in ("cargar_meta", "info_sal", "voltaje_termico", "cargar_malla"):
        assert callable(getattr(pc, fn, None)), f"missing {fn}()"
