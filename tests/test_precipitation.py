# -*- coding: utf-8 -*-
"""
Precipitation analyses of the extractor (Pillar 3, v1.1.0).

Pure NumPy/matplotlib checks plus the batch launcher run in-process on the
`fenics_stub` mesh (conftest), so they need neither FEniCS nor a solver run
(not marked `slow`). They pin:

  * the saturation Omega = Q_act/Ksp and its three classes agree with the
    binary map of v1.0.0 (class 2 <=> "precipitates");
  * voltaje_umbral() interpolates the onset voltage of each polarity;
  * the three map styles draw, and keep the v1.0.0 file name for "binario";
  * precipitation_onset: sweep, figure, saved files;
  * launch_extractor --opt: parsing, validation against guardar(), and a batch
    run of both modules with options.
"""
import json
import pathlib
import shutil
import sys

import pytest

np = pytest.importorskip("numpy")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.tri as mtri  # noqa: E402

_PRECIP_DIR = (pathlib.Path(__file__).resolve().parent.parent
               / "3_extractor" / "modulos" / "precipitation_map")
if str(_PRECIP_DIR) not in sys.path:
    sys.path.insert(0, str(_PRECIP_DIR))

import porosim_comun as pc  # noqa: E402  (3_extractor/modulos on sys.path via conftest)
import precipitation as pr  # noqa: E402
import precipitation_onset as po  # noqa: E402
import launch_extractor as le  # noqa: E402  (3_extractor on sys.path via conftest)

_DEMO = "RESULTS/solutions/demo-conical_KClO4_100.0mM"
_STEM = "demo-conical_KClO4_100.0mM"
_KSP = 0.0107          # KClO4, the demo's salt


@pytest.fixture(scope="module")
def meta_demo(repo_root):
    return json.loads((repo_root / _DEMO / f"Solutions_{_STEM}_sim.json")
                      .read_text(encoding="utf-8"))


@pytest.fixture
def datos_sinteticos():
    """A small mesh with a concentration hot spot at the tip: Omega spans
    under-saturated, supersaturated and precipitating nodes."""
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.uniform(0, 1300, 2500), np.linspace(0, 1300, 100)])
    r = np.concatenate([rng.uniform(0, 500, 2500), np.zeros(100)])
    tri = mtri.Triangulation(z, r).triangles
    pico = np.exp(-((z - 500) / 120) ** 2 - (r / 120) ** 2)
    c = 100.0 * (1.0 + 2.5 * pico)                 # 100 mM → 350 mM at the spot
    return {"z_nm": z, "r_nm": r, "tri": tri, "cp_mM": c, "cm_mM": c.copy()}


# ── physics ─────────────────────────────────────────────────────────────────
def test_classes_agree_with_the_binary_map(datos_sinteticos):
    d = datos_sinteticos
    for factor in (1.0, 2.0, 3.5):
        res = pr.calcular_precipitacion(d["cp_mM"], d["cm_mM"], 1, -1, _KSP, factor)
        clases = pr.clasificar_saturacion(d["cp_mM"], d["cm_mM"], 1, -1, _KSP, factor)
        omega = pr.saturacion(d["cp_mM"], d["cm_mM"], 1, -1, _KSP)
        np.testing.assert_array_equal(clases == 2, res["mapa_bin"] == 1)
        np.testing.assert_array_equal(clases >= 1, omega > 1)
        np.testing.assert_allclose(omega, res["Q_act"] / _KSP)
    assert set(np.unique(pr.clasificar_saturacion(
        d["cp_mM"], d["cm_mM"], 1, -1, _KSP, 2.0))) == {0.0, 1.0, 2.0}
    assert pr.s_max(d, _KSP, 1, -1) == pytest.approx(float(omega.max()))


def test_onset_voltage_is_interpolated_per_polarity():
    V = np.round(np.linspace(-1.0, 1.0, 21), 10)
    smax = np.where(V >= 0, 1.0 + 4.0 * V, 1.0 + 1.5 * np.abs(V))   # linear
    v_on = pr.voltaje_umbral(V, smax, 2.0)
    assert v_on["+"] == pytest.approx(0.25)
    assert v_on["-"] == pytest.approx(1.0 / 1.5)
    assert pr.voltaje_umbral(V, smax, 100.0) == {"+": None, "-": None}
    assert pr.voltaje_umbral(V, smax, 0.5) == {"+": 0.0, "-": 0.0}
    # the order of the input does not matter
    perm = np.random.default_rng(0).permutation(len(V))
    assert pr.voltaje_umbral(V[perm], smax[perm], 2.0) == pytest.approx(v_on)


def test_style_names_and_file_names():
    assert [pr.estilo_valido(e) for e in ("binary", "Ternary", "log", "log10")] == \
        ["binario", "ternario", "log10", "log10"]
    with pytest.raises(ValueError):
        pr.estilo_valido("rainbow")
    assert pr.nombre_base("S", "+1.00", 2.0) == "S_precipitation_+1.00V_f2"   # v1.0.0
    assert pr.nombre_base("S", "-0.50", 10, "ternary") == "S_saturation_ternary_-0.50V_f10"
    assert pr.nombre_base("S", "+0.00", 1.5, "log10") == "S_saturation_log10_+0.00V_f1.5"


@pytest.mark.parametrize("estilo, n_ejes", [("binario", 1), ("ternario", 1), ("log10", 2)])
def test_every_style_draws(datos_sinteticos, meta_demo, estilo, n_ejes):
    ctx = pc.contexto_de(meta_demo, _STEM, "+1.00")
    fig = pr.crear_figura(datos_sinteticos, ctx, factor=2.0, estilo=estilo)
    info = fig.info_precipitacion
    assert len(fig.axes) == n_ejes                 # log10 adds its colorbar
    assert 0 < info["n_precip"] < info["n_sobresat"] < info["n_total"]
    assert info["omega_max"] > 2.0
    mpl.pyplot.close(fig)


def test_guardar_names_and_counts(datos_sinteticos, meta_demo, tmp_path):
    ctx = pc.contexto_de(meta_demo, _STEM, "+1.00")
    ruta = tmp_path / f"Solutions_{_STEM}.h5"
    rutas = pr.guardar(datos_sinteticos, ctx, ruta, con_datos=True, factor=10,
                       estilo="ternary")
    nombres = sorted(p.name for p in rutas)
    assert nombres == [f"Solutions_{_STEM}_precipitation_+1.00V_f10.txt",
                       f"Solutions_{_STEM}_saturation_ternary_+1.00V_f10.png"]
    txt = (tmp_path / nombres[0]).read_text(encoding="utf-8")
    assert "factor_x_Kps\t10" in txt and "nodos_sobresaturados" in txt
    with pytest.raises(ValueError):
        pr.guardar(datos_sinteticos, ctx, ruta, factor=0)


# ── onset vs voltage ────────────────────────────────────────────────────────
def test_onset_module_sweep_figure_and_files(meta_demo, tmp_path):
    assert po.USA_VOLTAJE is False and po.aplica(meta_demo)

    def leer(lab):                                # c grows with |V|, faster at +V
        v = float(lab)
        u = np.full(50, np.log1p(3.0 * v if v > 0 else 1.0 * abs(v)))
        return {"up": u, "um": u}

    labels = [f"{v:+.2f}" for v in np.linspace(-1, 1, 11)]
    datos = po.calcular_barrido(leer, labels, 100.0, _KSP, 1, -1)
    assert list(datos["V"]) == sorted(float(l) for l in labels)
    assert np.all(np.diff(datos["S_max"][datos["V"] >= 0]) > 0)

    ctx = pc.contexto_de(meta_demo, _STEM)
    fig = po.crear_figura(datos, ctx, factor=2.0, v_read=0.4)
    esperado = pr.voltaje_umbral(datos["V"], datos["S_max"], 2.0)
    assert fig.onset == esperado and esperado["+"] < esperado["-"]
    mpl.pyplot.close(fig)

    ruta = tmp_path / f"Solutions_{_STEM}.h5"
    rutas = po.guardar(datos, ctx, ruta, con_datos=True, factor=1.5)
    assert sorted(p.name for p in rutas) == [
        f"Solutions_{_STEM}_precipitation_onset_f1.5.png",
        f"Solutions_{_STEM}_precipitation_onset_f1.5.txt"]
    tabla = np.loadtxt(tmp_path / f"Solutions_{_STEM}_precipitation_onset_f1.5.txt")
    np.testing.assert_allclose(tabla[:, 1], datos["S_max"], rtol=1e-6)


def test_onset_does_not_apply_to_soluble_or_mixture(meta_demo):
    soluble = json.loads(json.dumps(meta_demo))
    soluble["simulation"]["salt"].update(soluble=True, Ksp_M2=None)
    mezcla = json.loads(json.dumps(meta_demo))
    mezcla["simulation"].update(salt=None, c0_mM=None, ions=[
        {"symbol": "K⁺", "z": 1, "D_m2s": 1.96e-9, "c0_mM": 100.0},
        {"symbol": "Cl⁻", "z": -1, "D_m2s": 2.03e-9, "c0_mM": 110.0},
        {"symbol": "H⁺", "z": 1, "D_m2s": 9.31e-9, "c0_mM": 10.0}])
    assert not po.aplica(soluble) and not po.aplica(mezcla)


# ── launch_extractor --opt ──────────────────────────────────────────────────
def test_opt_values_are_parsed():
    assert le.parsear_opts(["factor=10", "estilo=ternary", "v_read=0.4",
                            "titulo=false", "ylim=-500,500", "x=none"]) == {
        "factor": 10, "estilo": "ternary", "v_read": 0.4, "titulo": False,
        "ylim": (-500, 500), "x": None}
    for malo in ("factor", "=3", "factor=", "2x=1"):
        with pytest.raises(SystemExit):
            le.parsear_opts([malo])


def test_opts_are_validated_against_guardar():
    assert le.opciones_modulo(pr) == {"factor": pr.FACTOR_PAPER, "estilo": "binario"}
    assert le.opciones_modulo(po) == {"factor": pr.FACTOR_PAPER, "v_read": None}
    le.validar_opts("precipitation", pr, {"factor": 10})
    with pytest.raises(SystemExit, match="Its options: factor"):
        le.validar_opts("precipitation", pr, {"colour": "red"})


def _carpeta_demo(repo_root, tmp_path):
    """Copy of the demo's _sim.json + I-V table and an (unread) empty .h5."""
    demo = repo_root / _DEMO
    for f in (f"Solutions_{_STEM}_sim.json", f"IV_curve_{_STEM}.txt"):
        shutil.copy(demo / f, tmp_path / f)
    h5 = tmp_path / f"Solutions_{_STEM}.h5"
    h5.write_bytes(b"")
    return h5


def test_batch_runs_both_modules_with_options(repo_root, fenics_stub, tmp_path):
    h5 = _carpeta_demo(repo_root, tmp_path)
    le._modo_batch(str(h5), "precipitation", "0.1", True,
                   {"factor": 0.5, "estilo": "log10"})
    le._modo_batch(str(h5), "precipitation_onset", None, True, {"factor": 0.5})
    nombres = {p.name for p in tmp_path.iterdir()}
    for esperado in (f"Solutions_{_STEM}_saturation_log10_+0.10V_f0.5.png",
                     f"Solutions_{_STEM}_precipitation_+0.10V_f0.5.txt",
                     f"Solutions_{_STEM}_precipitation_onset_f0.5.png",
                     f"Solutions_{_STEM}_precipitation_onset_f0.5.txt"):
        assert esperado in nombres, sorted(nombres)
    with pytest.raises(SystemExit, match="must be > 0"):
        le._modo_batch(str(h5), "precipitation", "0.1", False, {"factor": 0})
