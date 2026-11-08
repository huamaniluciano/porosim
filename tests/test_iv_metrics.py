# -*- coding: utf-8 -*-
"""
I-V metrics of the extractor (iv_analysis/iv_metrics.py, v1.1.0).

Pure NumPy on synthetic curves with known answers and on the committed demo
I-V table, plus the batch launcher in-process: no FEniCS (the module only
reads IV_curve_*.txt), not marked `slow`. They pin:

  * the definitions of the paper's analysis (referee response): n fitted on
    ln|I| vs ln|V| in 0.7-1 V, S-index = G(1 V)/max G, f_rec, G0 on |V| <= 0.1 V,
    checked against those formulas on the demo curve;
  * exact answers on power laws, an ohmic resistor and a saturating curve;
  * None (not a crash) when a branch or the fit window is missing;
  * the files written by guardar() and the batch with --opt.
"""
import pathlib
import shutil
import sys

import pytest

np = pytest.importorskip("numpy")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

_IV_DIR = (pathlib.Path(__file__).resolve().parent.parent
           / "3_extractor" / "modulos" / "iv_analysis")
if str(_IV_DIR) not in sys.path:
    sys.path.insert(0, str(_IV_DIR))

import iv_metrics as ivm  # noqa: E402
import launch_extractor as le  # noqa: E402  (3_extractor on sys.path via conftest)

_DEMO = "RESULTS/solutions/demo-conical_KClO4_100.0mM"
_STEM = "demo-conical_KClO4_100.0mM"
_V = np.round(np.linspace(-1.0, 1.0, 21), 10)


def _ley_potencia(n_pos, n_neg):
    return np.where(_V >= 0, 3.0 * np.abs(_V) ** n_pos, -2.0 * np.abs(_V) ** n_neg)


def test_power_law_gives_its_exponent_everywhere():
    m = ivm.metricas(_V, _ley_potencia(1.5, 0.8))
    assert m["n_pos"] == pytest.approx(1.5) and m["n_neg"] == pytest.approx(0.8)
    assert m["f_rec"] == pytest.approx(1.5)                    # 3 / 2 at |V| = 1
    pend = ivm.pendiente_local(_V, _ley_potencia(1.5, 0.8))
    np.testing.assert_allclose(pend["+"][1], 1.5)
    np.testing.assert_allclose(pend["-"][1], 0.8)
    # superlinear branch: G grows with V → no S shape; sublinear: G falls
    assert m["S_index_pos"] == pytest.approx(1.0)
    assert m["S_index_neg"] == pytest.approx(0.1 ** 0.2)        # G(1)/G(0.1)


def test_ohmic_resistor():
    m = ivm.metricas(_V, 4.0 * _V)
    assert (m["n_pos"], m["n_neg"], m["f_rec"]) == pytest.approx((1.0, 1.0, 1.0))
    assert m["G0_nS"] == pytest.approx(4.0)
    assert (m["S_index_pos"], m["S_index_neg"]) == pytest.approx((1.0, 1.0))


def test_saturating_branch_has_an_s_shape():
    I = np.where(_V >= 0, 5.0 * np.tanh(2.0 * _V), 4.0 * _V)
    m = ivm.metricas(_V, I)
    assert m["n_pos"] < 0.5 and m["S_index_pos"] < 0.6
    assert m["n_neg"] == pytest.approx(1.0)


def test_matches_the_paper_formulas_on_the_demo(repo_root):
    """Same numbers as metricas() of REFERATO/scripts/analizar_posicion.py."""
    d = ivm.cargar_iv(repo_root / _DEMO / f"IV_curve_{_STEM}.txt")
    V, I = d["V"], d["I"]
    np.testing.assert_allclose(I, 0.5 * (d["I_in"] + d["I_out"]))
    en = lambda v: I[np.argmin(abs(V - v))]                   # noqa: E731
    pos = V > 0.025
    G = I[pos] / V[pos]
    bajo = np.abs(V) <= 0.1 + 1e-9
    ventana = (V >= 0.7 - 1e-9) & (V <= 1.0 + 1e-9)
    m = ivm.metricas(V, I)
    assert m["S_index_pos"] == pytest.approx(G[-1] / G.max())
    assert m["f_rec"] == pytest.approx(abs(en(1.0) / en(-1.0)))
    assert m["G0_nS"] == pytest.approx(np.polyfit(V[bajo], I[bajo], 1)[0])
    assert m["n_pos"] == pytest.approx(
        np.polyfit(np.log(V[ventana]), np.log(I[ventana]), 1)[0])


def test_missing_branch_or_window_gives_none():
    V = np.round(np.linspace(0.0, 0.5, 6), 10)
    m = ivm.metricas(V, 2.0 * V)
    assert m["n_pos"] is None and m["n_neg"] is None            # no point in 0.7-1 V
    assert m["f_rec"] is None and m["S_index_neg"] is None
    assert m["S_index_pos"] == pytest.approx(1.0)
    assert all("n/a" in linea for linea in ivm.texto_metricas(m)
               if linea.startswith(("n_", "f_rec")))
    for malo in ({"ventana_n": (1.0, 0.7)}, {"ventana_n": (0.7,)}, {"v_ref": 0},
                 {"v_g0": -0.1}):
        with pytest.raises(ValueError):
            ivm.metricas(_V, 4.0 * _V, **malo)


def test_guardar_and_batch_with_options(repo_root, tmp_path):
    demo = repo_root / _DEMO
    for f in (f"Solutions_{_STEM}_sim.json", f"IV_curve_{_STEM}.txt"):
        shutil.copy(demo / f, tmp_path / f)
    h5 = tmp_path / f"Solutions_{_STEM}.h5"
    h5.write_bytes(b"")                                       # the module never reads it
    le._modo_batch(str(h5), "iv_metrics", None, True,
                   {"v_ref": 0.5, "ventana_n": (0.3, 0.5)})
    stem = f"Solutions_{_STEM}"
    for nombre in (f"{stem}_IV_metrics.txt", f"{stem}_IV_metrics.png",
                   f"{stem}_IV_local_slope.txt"):
        assert (tmp_path / nombre).stat().st_size > 0, nombre
    txt = (tmp_path / f"{stem}_IV_metrics.txt").read_text(encoding="utf-8")
    assert "|I(+0.50 V) / I(-0.50 V)|" in txt and "fitted on 0.3-0.5 V" in txt
    pendiente = np.loadtxt(tmp_path / f"{stem}_IV_local_slope.txt")
    assert pendiente.shape == (20, 2)                         # 21 voltages minus V = 0
    with pytest.raises(SystemExit, match="ventana_n"):
        le._modo_batch(str(h5), "iv_metrics", None, False, {"ventana_n": (0.9, 0.2)})
