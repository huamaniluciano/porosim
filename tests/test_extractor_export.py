# -*- coding: utf-8 -*-
"""
Figure export options of the extractor (Pillar 3, v1.1.0).

Pure NumPy/matplotlib checks plus the batch launcher run in-process on the
`fenics_stub` mesh (conftest): no FEniCS, not marked `slow`. They pin:

  * F9 (paper copy): the optional FIXED potential scale gives the same color
    to the same voltage in two solutions with different ranges, keeps 0 V
    white, honours gamma, and warns instead of saturating silently; the
    default (adaptive) scale is unchanged;
  * vector exports keep their text editable (TrueType in PDF, <text> in SVG);
  * `--format pdf` in batch, together with `--opt phi_ref=...`.
"""
import json
import shutil

import pytest

np = pytest.importorskip("numpy")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import porosim_comun as pc  # noqa: E402  (3_extractor/modulos on sys.path via conftest)
import launch_extractor as le  # noqa: E402  (3_extractor on sys.path via conftest)

_DEMO = "RESULTS/solutions/demo-conical_KClO4_100.0mM"
_STEM = "demo-conical_KClO4_100.0mM"


def test_fixed_scale_same_voltage_same_color(capsys):
    corrida_a = np.linspace(-0.25, 0.30, 50)
    corrida_b = np.linspace(-0.90, 0.10, 50)
    norm_a, niv_a = pc.escala_potencial(corrida_a, phi_ref=1.0)
    norm_b, niv_b = pc.escala_potencial(corrida_b, phi_ref=1.0)
    for v in (-0.2, 0.0, 0.05):
        assert float(norm_a(v)) == pytest.approx(float(norm_b(v)))
    assert float(norm_a(0.0)) == pytest.approx(0.5)           # 0 V = white
    # the levels stay the data range (honest colorbar), only the color is fixed
    assert niv_a[0] <= corrida_a.min() and niv_a[-1] >= corrida_a.max()
    assert niv_b[0] <= corrida_b.min() and niv_b[-1] >= corrida_b.max()
    assert "exceeds" not in capsys.readouterr().out

    pc.escala_potencial(corrida_b, phi_ref=0.5)                # |phi|max 0.9 > 0.5
    assert "exceeds the fixed scale ceiling" in capsys.readouterr().out


def test_fixed_scale_gamma_and_validation():
    norm, _ = pc.escala_potencial(np.linspace(-1, 1, 11), phi_ref=1.0, gamma=0.5)
    assert float(norm(0.25)) == pytest.approx(0.5 + 0.5 * 0.25 ** 0.5)
    assert float(norm(-0.25)) == pytest.approx(0.5 - 0.5 * 0.25 ** 0.5)
    for malo in ({"phi_ref": 0.0}, {"phi_ref": -1.0}, {"phi_ref": 1.0, "gamma": 0.0}):
        with pytest.raises(ValueError):
            pc.escala_potencial(np.linspace(-1, 1, 11), **malo)


def test_default_scale_is_still_adaptive():
    phi = np.linspace(-0.3, 0.6, 40)
    norm, niveles = pc.escala_potencial(phi)
    assert isinstance(norm, mcolors.TwoSlopeNorm)
    assert (norm.vmin, norm.vcenter, norm.vmax) == pytest.approx((-0.3, 0.0, 0.6))
    assert niveles[0] == pytest.approx(-0.3) and niveles[-1] == pytest.approx(0.6)


@pytest.fixture
def formato_png():
    """Restore the global output format after the test."""
    yield
    pc.fijar_formato("png")


def test_vector_exports_keep_editable_text(tmp_path, formato_png):
    with pytest.raises(ValueError):
        pc.fijar_formato("bmp")
    for fmt in ("pdf", "svg"):
        pc.fijar_formato(fmt)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title("Potential map")
        ruta = pc.guardar_figura(fig, tmp_path / "Solutions_x.h5", "Solutions_x_map.png")
        assert ruta.name == f"Solutions_x_map.{fmt}"
        contenido = ruta.read_bytes()
        if fmt == "pdf":
            # TrueType outlines embedded (FontFile2), never Type 3 glyph procedures
            assert b"/FontFile2" in contenido and b"/Type3" not in contenido
        else:
            assert b"<text" in contenido and b"Potential map" in contenido


def test_batch_format_and_fixed_scale(repo_root, fenics_stub, tmp_path, formato_png):
    demo = repo_root / _DEMO
    for f in (f"Solutions_{_STEM}_sim.json", f"IV_curve_{_STEM}.txt"):
        shutil.copy(demo / f, tmp_path / f)
    h5 = tmp_path / f"Solutions_{_STEM}.h5"
    h5.write_bytes(b"")
    le._modo_batch(str(h5), "potential", "0.1", False,
                   {"phi_ref": 2.0, "gamma": 0.7}, "pdf")
    le._modo_batch(str(h5), "solution_summary", None, False, {}, "pdf")
    nombres = {p.name for p in tmp_path.iterdir()}
    for esperado in (f"Solutions_{_STEM}_potential_map_+0.10V.pdf",
                     f"Solutions_{_STEM}_IV_curve.pdf",
                     f"Solutions_{_STEM}_channel_schematic.pdf"):
        assert esperado in nombres, sorted(nombres)
    assert not any(n.endswith(".png") for n in nombres)
    with pytest.raises(SystemExit, match="--format"):
        le._modo_batch(str(h5), "potential", "0.1", False, {}, "bmp")
    with pytest.raises(SystemExit, match="must be > 0"):
        le._modo_batch(str(h5), "potential", "0.1", False, {"phi_ref": -1}, "png")


def test_meta_demo_is_binary(repo_root):
    """Guard: the demo used above is the binary KClO4 run of v1.0.0."""
    meta = json.loads((repo_root / _DEMO / f"Solutions_{_STEM}_sim.json")
                      .read_text(encoding="utf-8"))
    assert pc.es_binaria(meta["simulation"])
