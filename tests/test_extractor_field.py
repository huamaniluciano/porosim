# -*- coding: utf-8 -*-
"""
Electric-field and figure helpers of the extractor (Pillar 3, added in v1.1.0).

Pure NumPy/matplotlib checks on a synthetic axisymmetric field, so they run in
seconds and need neither FEniCS nor a solver run (not marked `slow`). They pin
the fixes ported from the paper copies (ARREGLOS PARA POROSIM/PORTABLE_FIXES.md):

  F3  Er = 0 on the symmetry axis and an exactly odd Er(r)
  F5  a zoom window gets its own field grid (it used to contain ~1 node)
  F4  zoomed field lines are drawn in exact mirror pairs
  F7  without zoom the axes end at the data, with no autoscale padding
  F2  colorbar ticks never fall outside the bar
  F8  limpiar_figura() keeps the frame and the tick marks
"""
import json
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_POT_DIR = pathlib.Path(__file__).resolve().parent.parent / "3_extractor" / "modulos" / "potential_map"
if str(_POT_DIR) not in sys.path:
    sys.path.insert(0, str(_POT_DIR))

import porosim_comun as pc  # noqa: E402  (3_extractor/modulos is on sys.path via conftest)
import potential  # noqa: E402

_DEMO_SIM = ("RESULTS/solutions/demo-conical_KClO4_100.0mM/"
             "Solutions_demo-conical_KClO4_100.0mM_sim.json")


def _campo_sintetico(z_max=1300.0, r_max=500.0, foco=(500.0, 0.0), n=6000,
                     n_foco=3000, ancho_foco=60.0, seed=0):
    """Unstructured mesh on r >= 0 (denser around `foco`) and an axisymmetric
    potential phi(z, r) = 0.3 tanh((z - z_foco)/40) exp(-r^2 / 150^2): a
    function of r^2, so the exact dphi/dr vanishes on the axis."""
    import matplotlib.tri as mtri
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, z_max, n)
    r = rng.uniform(0.0, r_max, n)
    zf = rng.uniform(foco[0] - ancho_foco, foco[0] + ancho_foco, n_foco)
    rf = rng.uniform(0.0, ancho_foco, n_foco)
    # nodes exactly on the axis and on the outer edges, like a real mesh
    z_eje = np.linspace(0.0, z_max, 400)
    z = np.concatenate([z, zf, z_eje, z_eje, np.zeros(50), np.full(50, z_max)])
    r = np.concatenate([r, rf, np.zeros_like(z_eje), np.full_like(z_eje, r_max),
                        np.linspace(0, r_max, 50), np.linspace(0, r_max, 50)])
    tri = mtri.Triangulation(z, r).triangles
    phi = 0.3 * np.tanh((z - foco[0]) / 40.0) * np.exp(-(r / 150.0) ** 2)
    return {"z_nm": z, "r_nm": r, "tri": tri, "phi_V": phi}


def test_er_is_zero_on_axis_and_odd():
    d = _campo_sintetico()
    ec = potential.campo_electrico(d["z_nm"], d["r_nm"], d["tri"], d["phi_V"])
    R, Er = ec["R_full"], ec["Er"]
    eje = np.flatnonzero(np.isclose(R, 0.0))
    assert eje.size == 1, "the mirrored grid must hold exactly one r = 0 row"
    fila = Er[eje[0]]
    assert np.isfinite(fila).any()
    assert np.all(fila[np.isfinite(fila)] == 0.0), "Er must vanish on the axis"
    # exactly odd in r: Er(-r) = -Er(r), NaNs in the same places
    espejo = -Er[::-1]
    assert np.array_equal(np.isnan(Er), np.isnan(espejo))
    finitos = np.isfinite(Er)
    assert np.array_equal(Er[finitos], espejo[finitos])


def test_zoom_window_gets_its_own_grid():
    d = _campo_sintetico(z_max=22000.0, r_max=5000.0, foco=(5000.0, 0.0),
                         n=20000, n_foco=4000)
    zlim, rlim = (4950.0, 5050.0), (-50.0, 50.0)

    def nodos_en_ventana(ec):
        Zg, Rg = np.meshgrid(ec["Z"], ec["R_full"])
        dentro = ((Zg >= zlim[0]) & (Zg <= zlim[1]) &
                  (Rg >= rlim[0]) & (Rg <= rlim[1]) & np.isfinite(ec["Ez"]))
        return int(dentro.sum())

    completo = potential.campo_electrico(d["z_nm"], d["r_nm"], d["tri"], d["phi_V"])
    zoom = potential.campo_electrico(d["z_nm"], d["r_nm"], d["tri"], d["phi_V"],
                                     zlim=zlim, rlim=rlim)
    assert nodos_en_ventana(completo) <= 4      # the v1.0.0 behaviour
    assert nodos_en_ventana(zoom) > 1000
    assert zoom["Z"].min() >= zlim[0] and zoom["Z"].max() <= zlim[1]
    assert np.abs(zoom["R_full"]).max() <= rlim[1] + 1e-9


def test_zoom_window_is_clamped_to_the_domain():
    d = _campo_sintetico()
    ec = potential.campo_electrico(d["z_nm"], d["r_nm"], d["tri"], d["phi_V"],
                                   zlim=(-200.0, 100.0), rlim=(-900.0, 900.0))
    assert ec["Z"].min() >= d["z_nm"].min()
    assert np.abs(ec["R_full"]).max() <= d["r_nm"].max()


def test_colorbar_ticks_stay_inside_the_bar():
    fig, ax = plt.subplots()
    im = ax.imshow([[-0.0003, 0.3098]], vmin=-0.0003, vmax=0.3098)
    cbar = fig.colorbar(im, ax=ax)
    ticks = pc.ticks_colorbar(cbar, -0.0003, 0.3098)
    plt.close(fig)
    assert len(ticks) >= 5
    assert ticks.min() >= -0.0003 and ticks.max() <= 0.3098
    assert np.any(np.isclose(ticks, 0.0))


def test_limpiar_figura_keeps_frame_and_tick_marks():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="x")
    ax.legend()
    ax.set_title("t")
    ax.set_xlabel("z")
    pc.limpiar_figura(fig)
    fig.canvas.draw()
    assert all(s.get_visible() for s in ax.spines.values())
    assert ax.get_legend() is None and ax.get_title() == "" and ax.get_xlabel() == ""
    assert len(ax.get_xticks()) > 0
    assert all(t.get_text() == "" for t in ax.get_xticklabels())
    plt.close(fig)


@pytest.fixture(scope="module")
def ctx_demo(repo_root):
    meta = json.loads((repo_root / _DEMO_SIM).read_text(encoding="utf-8"))
    return pc.contexto_de(meta, "demo-conical_KClO4_100.0mM", "+0.40")


def test_full_view_ends_at_the_data(ctx_demo):
    d = _campo_sintetico()
    fig = potential.crear_figura(d, ctx_demo, con_campo=True)
    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((d["z_nm"].min(), d["z_nm"].max()))
    assert ax.get_ylim() == pytest.approx((-d["r_nm"].max(), d["r_nm"].max()))
    plt.close(fig)


def test_zoomed_field_lines_come_in_mirror_pairs(ctx_demo):
    from matplotlib.collections import LineCollection
    d = _campo_sintetico()
    fig = potential.crear_figura(d, ctx_demo, con_campo=True,
                                 xlim=(440.0, 560.0), ylim=(-50.0, 50.0))
    ax = fig.axes[0]
    colecciones = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert colecciones, "zoomed figure has no field lines"
    segmentos = [s for c in colecciones for s in c.get_segments() if len(s) > 1]
    assert len(segmentos) >= 4

    def firma(v):
        return tuple(np.round(v, 6).ravel())

    firmas = {firma(v) for v in segmentos}
    for v in segmentos:
        if np.abs(v[:, 1]).max() <= 1e-9:     # the line along the axis is not doubled
            continue
        espejada = v.copy()
        espejada[:, 1] *= -1.0
        assert firma(espejada) in firmas, "a field line has no mirror image"
    plt.close(fig)
