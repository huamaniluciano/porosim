# -*- coding: utf-8 -*-
"""
Extractor GUI (gui_extractor_app.py) driven with Streamlit's AppTest.

The FEniCS-dependent loaders of porosim_comun (mesh, voltages, nodal fields)
are replaced by a synthetic axisymmetric mesh and fields, so the test runs in
seconds without FEniCS (not marked `slow`). It walks every per-voltage view,
presses every "Compute and Plot" button and fails on any exception, for:

  * the committed binary demo (KClO4: four views, precipitation included);
  * a mixture (KCl + HCl, three ions, v1.1.0): only the potential views, and
    the axial profile offers the potential alone.
"""
import json
import shutil

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
from streamlit.testing.v1 import AppTest  # noqa: E402

import porosim_comun as pc  # noqa: E402  (3_extractor/modulos on sys.path via conftest)

_DEMO = "RESULTS/solutions/demo-conical_KClO4_100.0mM"
_STEM_DEMO = "demo-conical_KClO4_100.0mM"
_STEM_MIX = "demo-conical_KCl_100mM+HCl_10mM"


@pytest.fixture
def fenics_stub(monkeypatch):
    """Synthetic mesh + fields in place of the .h5 readers (FEniCS)."""
    import matplotlib.tri as mtri
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.uniform(0, 1300e-9, 3000), np.linspace(0, 1300e-9, 200)])
    r = np.concatenate([rng.uniform(0, 500e-9, 3000), np.zeros(200)])
    tri = mtri.Triangulation(z, r).triangles

    def datos_malla(_mesh):
        eje = np.where(r < 1e-12)[0]
        return {"z_nm": z * 1e9, "r_nm": r * 1e9, "tri": tri,
                "axis_idx": eje[np.argsort(z[eje])]}

    def leer_campos(_ruta, _mesh, V, _v_label):
        n_iones = V[1]
        phi = 0.3 * np.tanh((z * 1e9 - 500) / 40) * np.exp(-(r * 1e9 / 150) ** 2) / 0.0257
        campos = {"phi_adim": phi,
                  "us": [0.1 * np.sin(z * 1e7 + k) for k in range(n_iones)]}
        if n_iones == 2:
            campos["up"], campos["um"] = campos["us"]
        return campos

    monkeypatch.setattr(pc, "cargar_malla", lambda _h5: object())
    monkeypatch.setattr(pc, "espacio_mixto", lambda _mesh, n_iones=2: ("V", n_iones))
    monkeypatch.setattr(pc, "datos_malla", datos_malla)
    monkeypatch.setattr(pc, "detectar_voltajes",
                        lambda _h5, _mesh, sim=None: ["-0.10", "+0.00", "+0.10"])
    monkeypatch.setattr(pc, "leer_campos", leer_campos)


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
