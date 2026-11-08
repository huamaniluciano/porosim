# -*- coding: utf-8 -*-
"""
Electrolyte → ions (v1.1.0): mixtures of salts and asymmetric salts.

Fast checks, no FEniCS (not marked `slow`):

  * 2_solver/electrolito.py: stoichiometry, merging of shared ions, validation,
    file labels and the numerical Donnan potential;
  * the batch validation of solver.py for the new "electrolyte" block (the
    engine is replaced by a stub that records the cfg it receives);
  * the extractor's shared layer reading a mixture _sim.json, and the
    concentration modules declaring themselves not applicable to it.

The end-to-end mixture runs (FEniCS) are in test_solver_mixture.py.
"""
import copy
import json
import math
import sys
import types

import pytest

import electrolito as el      # 2_solver/ is on sys.path via conftest

_SALES = None


def _catalogo(repo_root):
    global _SALES
    if _SALES is None:
        _SALES = json.loads((repo_root / "2_solver" / "sales.json")
                            .read_text(encoding="utf-8"))["salts"]
    return copy.deepcopy(_SALES)


def _sal(nombre, z_p, z_m, D_p=1e-9, D_m=2e-9, sp=None, sm=None):
    return {"name": nombre,
            "cation": {"symbol": sp or f"{nombre}+", "z": z_p, "D_m2s": D_p},
            "anion":  {"symbol": sm or f"{nombre}-", "z": z_m, "D_m2s": D_m},
            "soluble": True, "Ksp_M2": None}


# ── stoichiometry ────────────────────────────────────────────────────────────
def test_symmetric_salt_keeps_c0_untouched(repo_root):
    kcl = _catalogo(repo_root)["KCl"]
    iones = el.iones_de_sal(kcl, 100.0)
    assert [i["c0_mM"] for i in iones] == [100.0, 100.0]
    assert [i["symbol"] for i in iones] == ["K⁺", "Cl⁻"]
    # same object and type as given (binary mode must not touch c0)
    assert type(el.iones_de_sal(kcl, 100)[0]["c0_mM"]) is int


@pytest.mark.parametrize("z_p, z_m, nu_p, nu_m", [(2, -1, 1, 2), (1, -2, 2, 1),
                                                  (2, -2, 1, 1), (3, -2, 2, 3)])
def test_neutral_stoichiometry(z_p, z_m, nu_p, nu_m):
    iones = el.iones_de_sal(_sal("X", z_p, z_m), 10.0)
    assert [i["c0_mM"] for i in iones] == [10.0 * nu_p, 10.0 * nu_m]
    assert z_p * iones[0]["c0_mM"] + z_m * iones[1]["c0_mM"] == 0


def test_bad_valences_are_rejected():
    with pytest.raises(ValueError, match="cation needs z > 0"):
        el.iones_de_sal(_sal("X", -1, 1), 10.0)
    with pytest.raises(ValueError, match="must be an integer"):
        el.iones_de_sal(_sal("X", 1.5, -1), 10.0)


# ── mixtures ────────────────────────────────────────────────────────────────
def test_kcl_plus_hcl_sums_the_chloride(repo_root):
    cat = _catalogo(repo_root)
    iones, info, etiqueta = el.combinar_sales([(cat["KCl"], 100), (cat["HCl"], 10)])
    assert [(i["symbol"], i["c0_mM"]) for i in iones] == \
        [("K⁺", 100.0), ("Cl⁻", 110.0), ("H⁺", 10.0)]
    assert info == [{"salt": "KCl", "c0_mM": 100.0}, {"salt": "HCl", "c0_mM": 10.0}]
    assert etiqueta == "KCl_100mM+HCl_10mM"


def test_kcl_plus_koh_sums_the_potassium(repo_root):
    cat = _catalogo(repo_root)
    iones, _, _ = el.combinar_sales([(cat["KCl"], 100), (cat["KOH"], 10)])
    assert [(i["symbol"], i["c0_mM"]) for i in iones] == \
        [("K⁺", 110.0), ("Cl⁻", 100.0), ("OH⁻", 10.0)]


def test_shared_ion_must_agree_in_z_and_D(repo_root):
    cat = _catalogo(repo_root)
    otro = copy.deepcopy(cat["HCl"])
    otro["name"], otro["anion"]["D_m2s"] = "HCl_bad", 1.0e-9
    with pytest.raises(ValueError, match="different z or D"):
        el.combinar_sales([(cat["KCl"], 100), (otro, 10)])


def test_non_positive_concentration_is_rejected(repo_root):
    cat = _catalogo(repo_root)
    with pytest.raises(ValueError, match="must be > 0"):
        el.combinar_sales([(cat["KCl"], 100), (cat["HCl"], 0)])


# ── iones_de_cfg: which mode the engine runs in ──────────────────────────────
def test_symmetric_salt_runs_in_binary_mode_with_v100_label(repo_root):
    iones, general, etiqueta = el.iones_de_cfg(
        {"salt": _catalogo(repo_root)["KCl"], "c0_mM": 100.0})
    assert general is False
    assert etiqueta == "KCl_100.0mM"            # Solutions_<mesh>_KCl_100.0mM.h5


def test_asymmetric_salt_runs_in_general_mode():
    iones, general, etiqueta = el.iones_de_cfg({"salt": _sal("CaCl2", 2, -1), "c0_mM": 10.0})
    assert general is True
    assert [i["c0_mM"] for i in iones] == [10.0, 20.0]
    assert etiqueta == "CaCl2_10.0mM"


def test_ions_cfg_is_validated():
    base = [{"symbol": "K", "z": 1, "D_m2s": 2e-9, "c0_mM": 100.0},
            {"symbol": "Cl", "z": -1, "D_m2s": 2e-9, "c0_mM": 100.0}]
    iones, general, etiqueta = el.iones_de_cfg({"ions": base})
    assert general is True and etiqueta == "K100+Cl100mM"
    no_neutro = copy.deepcopy(base)
    no_neutro[1]["c0_mM"] = 90.0
    with pytest.raises(ValueError, match="not electroneutral"):
        el.iones_de_cfg({"ions": no_neutro})
    with pytest.raises(ValueError, match="At least 2 ions"):
        el.iones_de_cfg({"ions": base[:1]})
    repetido = copy.deepcopy(base)
    repetido[1]["symbol"] = "K"
    with pytest.raises(ValueError, match="own non-empty symbol"):
        el.iones_de_cfg({"ions": repetido})
    fraccion = copy.deepcopy(base)
    fraccion[0]["z"] = 1.5
    with pytest.raises(ValueError, match="non-zero integer"):
        el.iones_de_cfg({"ions": fraccion})


def test_ascii_labels_for_file_names():
    assert el._ascii("ClO₄⁻") == "ClO4"
    assert el._ascii("Ca²⁺") == "Ca2"


# ── Donnan ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("c_fix", [-4000.0, -50.0, 0.0, 3.0, 660.0])
def test_donnan_reduces_to_arcsinh_for_a_1_1_salt(repo_root, c_fix):
    iones = el.iones_de_sal(_catalogo(repo_root)["KCl"], 100.0)
    assert el.donnan_psi(iones, c_fix) == pytest.approx(
        math.asinh(c_fix / 200.0), rel=1e-12, abs=1e-14)


def test_donnan_of_a_mixture_is_electroneutral(repo_root):
    cat = _catalogo(repo_root)
    iones, _, _ = el.combinar_sales([(cat["KCl"], 100), (cat["HCl"], 10),
                                     (_sal("CaCl2", 2, -1, sp="Ca²⁺", sm="Cl⁻",
                                           D_m=2.03e-9), 5)])
    c_fix = -1000.0                        # negative film → φ_D < 0
    psi = el.donnan_psi(iones, c_fix)
    assert psi < 0
    carga = sum(i["z"] * i["c0_mM"] * math.exp(-i["z"] * psi) for i in iones) + c_fix
    assert abs(carga) < 1e-9 * abs(c_fix)


# ── solver.py batch validation (engine stubbed: no FEniCS) ────────────────────
@pytest.fixture
def batch(repo_root, tmp_path, monkeypatch):
    """Run solver._generar_batch on a params dict; returns the cfg the engine
    would receive (or raises SystemExit with the validation message)."""
    capturado = {}
    falso = types.ModuleType("motor_pnp")
    falso.resolver = lambda cfg: capturado.update(cfg)
    monkeypatch.setitem(sys.modules, "motor_pnp", falso)
    import solver

    def correr(params, mesh="RESULTS/meshes/demo-conical"):
        spec = {"mesh": str(repo_root / mesh), "output": str(tmp_path / "out"),
                "params": params}
        ruta = tmp_path / "params.json"
        ruta.write_text(json.dumps(spec), encoding="utf-8")
        solver._generar_batch(str(ruta))
        return dict(capturado)

    return correr


def test_batch_mixture_reaches_the_engine_as_ions(batch):
    cfg = batch({"electrolyte": [{"salt": "KCl", "c0_mM": 100},
                                 {"salt": "HCl", "c0_mM": 10}]})
    assert "salt" not in cfg and "c0_mM" not in cfg
    assert [(i["symbol"], i["c0_mM"]) for i in cfg["ions"]] == \
        [("K⁺", 100.0), ("Cl⁻", 110.0), ("H⁺", 10.0)]
    assert cfg["electrolyte_label"] == "KCl_100mM+HCl_10mM"
    assert cfg["electrolyte"][1] == {"salt": "HCl", "c0_mM": 10.0}


def test_batch_binary_cfg_is_unchanged(batch):
    cfg = batch({"salt": "KCl", "c0_mM": 100.0})
    assert cfg["salt"]["name"] == "KCl" and cfg["c0_mM"] == 100.0
    assert "ions" not in cfg and "electrolyte" not in cfg


@pytest.mark.parametrize("params, mensaje", [
    ({"electrolyte": [{"salt": "KCl", "c0_mM": 100}], "salt": "KCl"}, "replaces"),
    ({"electrolyte": []}, "non-empty list"),
    ({"electrolyte": [{"salt": "KCl"}]}, "must be"),
    ({"electrolyte": [{"salt": "XYZ", "c0_mM": 1}]}, "not in the catalog"),
    ({"electrolyte": [{"salt": "KCl", "c0_mM": -1}]}, "must be > 0"),
])
def test_batch_rejects_bad_electrolyte(batch, params, mensaje):
    with pytest.raises(SystemExit, match=mensaje):
        batch(params)


# ── extractor shared layer on a mixture _sim.json ─────────────────────────────
def _meta_mezcla():
    return {"z_tip": 0.0, "z_base": 1e-6, "R_tip": 5e-9, "R_base": 1e-7,
            "simulation": {
                "salt": None, "c0_mM": None, "c0_inlet_mM": None, "c0_outlet_mM": None,
                "T_K": 298.15, "sigma_Cm2": -0.16, "films": [],
                "ions": [{"symbol": "K⁺", "z": 1, "D_m2s": 1.96e-9, "c0_mM": 100.0},
                         {"symbol": "Cl⁻", "z": -1, "D_m2s": 2.03e-9, "c0_mM": 110.0},
                         {"symbol": "H⁺", "z": 1, "D_m2s": 9.31e-9, "c0_mM": 10.0}],
                "electrolyte": [{"salt": "KCl", "c0_mM": 100.0},
                                {"salt": "HCl", "c0_mM": 10.0}],
                "electrolyte_label": "KCl_100mM+HCl_10mM"}}


def test_extractor_context_of_a_mixture():
    import porosim_comun as pc
    meta = _meta_mezcla()
    ctx = pc.contexto_de(meta, "stem", "+0.10")
    assert ctx["c0"] is None and ctx["binaria"] is False
    assert len(ctx["ions"]) == 3
    assert ctx["salt"]["name"] == "KCl_100mM+HCl_10mM"
    assert ctx["salt"]["label_p"] == "K⁺ + H⁺" and ctx["salt"]["label_m"] == "Cl⁻"
    assert pc.describir_electrolito(meta["simulation"]) == \
        "KCl_100mM+HCl_10mM: K⁺ 100, Cl⁻ 110, H⁺ 10 mM"


def test_extractor_context_of_a_v100_binary_sim_json():
    import porosim_comun as pc
    sim = {"salt": {"name": "KCl", "cation": {"symbol": "K⁺", "z": 1, "D_m2s": 1.96e-9},
                    "anion": {"symbol": "Cl⁻", "z": -1, "D_m2s": 2.03e-9},
                    "soluble": True, "Ksp_M2": None},
           "c0_mM": 100.0}                       # no "ions": written by v1.0.0
    iones = pc.iones_de_sim(sim)
    assert [(i["symbol"], i["z"], i["c0_mM"]) for i in iones] == \
        [("K⁺", 1, 100.0), ("Cl⁻", -1, 100.0)]
    assert pc.es_binaria(sim)
    assert pc.contexto_de({"simulation": sim}, "s")["c0"] == 100.0


def test_concentration_modules_do_not_apply_to_a_mixture(repo_root):
    import importlib.util
    mods = repo_root / "3_extractor" / "modulos"
    meta_mezcla = _meta_mezcla()
    meta_kclo4 = {"simulation": {"c0_mM": 10.0, "salt": {
        "name": "KClO4", "cation": {"symbol": "K⁺", "z": 1, "D_m2s": 1.96e-9},
        "anion": {"symbol": "ClO₄⁻", "z": -1, "D_m2s": 1.79e-9},
        "soluble": False, "Ksp_M2": 1.07e-2}}}
    for rel in ("ion_maps/ions.py", "ion_maps/total_ions.py",
                "axial_profiles/axis_profile_ions.py",
                "axial_profiles/section_avg_concentration.py",
                "precipitation_map/precipitation.py"):
        ruta = mods / rel
        sys.path.insert(0, str(ruta.parent))
        try:
            spec = importlib.util.spec_from_file_location(f"_t_{ruta.stem}", ruta)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(ruta.parent))
        assert mod.aplica(meta_mezcla) is False, rel
        assert mod.aplica(meta_kclo4) is True, rel
