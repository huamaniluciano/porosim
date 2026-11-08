# -*- coding: utf-8 -*-
"""
Extractor GUI (gui_extractor_app.py) driven with Streamlit's AppTest.

The FEniCS-dependent loaders of porosim_comun (mesh, voltages, nodal fields)
are replaced by a synthetic axisymmetric mesh and fields (`fenics_stub`,
conftest), so the test runs in seconds without FEniCS (not marked `slow`). It
walks every per-voltage view, presses every "Compute and Plot" button and fails
on any exception, for:

  * the committed binary demo (KClO4: four views, precipitation included);
  * the same demo with the three precipitation map styles, a threshold S = 10
    and the onset-vs-voltage sweep of the full-sweep tab (v1.1.0);
  * the same demo with the fixed potential color scale, whose saturation
    warning must show up (v1.1.0);
  * a mixture (KCl + HCl, three ions, v1.1.0): only the potential views, and
    the axial profile offers the potential alone.
"""
import json
import shutil

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pandas")
pytest.importorskip("numpy")
from streamlit.testing.v1 import AppTest  # noqa: E402

_DEMO = "RESULTS/solutions/demo-conical_KClO4_100.0mM"
_STEM_DEMO = "demo-conical_KClO4_100.0mM"
_STEM_MIX = "demo-conical_KCl_100mM+HCl_10mM"


def _solucion_mezcla(repo_root, carpeta):
    """A mixture solution folder: the demo's _sim.json turned into
    KCl 100 + HCl 10 mM, its I-V table and an (unread) .h5."""
    demo = repo_root / _DEMO
    meta = json.loads((demo / f"Solutions_{_STEM_DEMO}_sim.json").read_text(encoding="utf-8"))
    meta["simulation"].update({
        "salt": None, "c0_mM": None, "c0_inlet_mM": None, "c0_outlet_mM": None,
        "ions": [{"symbol": "K⁺", "z": 1, "D_m2s": 1.96e-9, "c0_mM": 100.0},
                 {"symbol": "Cl⁻", "z": -1, "D_m2s": 2.03e-9, "c0_mM": 110.0},
                 {"symbol": "H⁺", "z": 1, "D_m2s": 9.31e-9, "c0_mM": 10.0}],
        "electrolyte": [{"salt": "KCl", "c0_mM": 100.0}, {"salt": "HCl", "c0_mM": 10.0}],
        "electrolyte_label": "KCl_100mM+HCl_10mM"})
    (carpeta / f"Solutions_{_STEM_MIX}_sim.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.copy(demo / f"IV_curve_{_STEM_DEMO}.txt", carpeta / f"IV_curve_{_STEM_MIX}.txt")
    h5 = carpeta / f"Solutions_{_STEM_MIX}.h5"
    h5.write_bytes(b"")
    return h5


def _abrir(repo_root, h5):
    at = AppTest.from_file(str(repo_root / "3_extractor" / "gui_extractor_app.py"),
                           default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].set_value("Enter path manually...").run()
    at.sidebar.text_input[0].set_value(str(h5)).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _recorrer_vistas(at):
    """Every per-voltage view, every compute button; returns the views and the
    axial-profile modes offered."""
    vistas = [x for x in at.radio if x.key == "vista_local"][0].options
    modos = None
    for vista in vistas:
        [x for x in at.radio if x.key == "vista_local"][0].set_value(vista).run()
        for boton in [b for b in at.button if "Compute and Plot" in str(b.label)]:
            boton.click().run()
            assert not at.exception, (vista, [e.message for e in at.exception])
        modo = [x for x in at.radio if "Visualization mode" in str(x.label)]
        if modo:
            modos = list(modo[0].options)
    return vistas, modos


def test_gui_binary_demo(repo_root, fenics_stub):
    at = _abrir(repo_root, repo_root / _DEMO / f"Solutions_{_STEM_DEMO}.h5")
    vistas, modos = _recorrer_vistas(at)
    assert len(vistas) == 4 and any("Precipitation" in v for v in vistas)
    assert len(modos) == 3


def test_gui_mixture(repo_root, fenics_stub, tmp_path):
    at = _abrir(repo_root, _solucion_mezcla(repo_root, tmp_path))
    texto = " ".join(m.value for m in at.sidebar.markdown)
    assert "KCl_100mM+HCl_10mM: K⁺ 100, Cl⁻ 110, H⁺ 10 mM" in texto
    vistas, modos = _recorrer_vistas(at)
    assert len(vistas) == 2 and not any("concentration" in v for v in vistas)
    assert modos == ["Potential only (φ(z))"]


def test_gui_precipitation_styles_and_onset(repo_root, fenics_stub):
    at = _abrir(repo_root, repo_root / _DEMO / f"Solutions_{_STEM_DEMO}.h5")

    # Full-sweep tab: onset vs voltage (reads every voltage of the stub)
    [b for b in at.button if "Compute sweep" in str(b.label)][0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    etiquetas = [str(m.label) for m in at.metric]
    assert any(e.startswith("Onset (+V) at S = 2") for e in etiquetas), etiquetas

    # Per-voltage precipitation view: every style, threshold S = 10
    vista = [v for v in [x for x in at.radio if x.key == "vista_local"][0].options
             if "Precipitation" in v][0]
    [x for x in at.radio if x.key == "vista_local"][0].set_value(vista).run()
    estilo = [x for x in at.radio if x.key == "precip__estilo"][0]
    assert len(estilo.options) == 3
    for opcion in estilo.options:
        [x for x in at.radio if x.key == "precip__estilo"][0].set_value(opcion)
        at.number_input(key="precip__S").set_value(10.0)
        [b for b in at.button if "Compute and Plot" in str(b.label)][0].click().run()
        assert not at.exception, (opcion, [e.message for e in at.exception])
        assert any("Supersaturated nodes" in str(m.label) for m in at.metric)


def test_gui_fixed_potential_scale(repo_root, fenics_stub):
    at = _abrir(repo_root, repo_root / _DEMO / f"Solutions_{_STEM_DEMO}.h5")
    vista = [v for v in [x for x in at.radio if x.key == "vista_local"][0].options
             if "potential" in v][0]
    [x for x in at.radio if x.key == "vista_local"][0].set_value(vista).run()
    at.checkbox(key="pot__fija").check()
    at.number_input(key="pot__phi_ref").set_value(0.05)
    at.number_input(key="pot__gamma").set_value(0.7)
    [b for b in at.button if "Compute and Plot" in str(b.label)][0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("fixed scale ceiling" in w.value for w in at.warning)
