# -*- coding: utf-8 -*-
"""
T2 - Mesher functional test (Pillar 1).

Actually runs the mesher on a small geometry (~5-7 s) and checks the contract
it hands to the Solver (Pillar 2): the exported files exist, <name>_limits.json
is well-formed, and the physical tags match the declared contract (TAGS in
capa1_modelo).

Two configurations are covered so the film branch is exercised:
  * "film_less": the small shipped example geom_conico_chico.json (no film).
  * "film_tip" : the same geometry with a tip film added, so the FILM_TIP_*
                 tags appear.

Each mesh is built ONCE (session-scoped, parametrized fixture); the individual
tests assert different aspects of that single output. Needs gmsh + meshio
(no FEniCS).

(POROSIM's own identifiers -- capa1_modelo, capa4_malla, mallar, Params, the
limits.json keys and TAGS -- are in Spanish and used verbatim: they are the
real contract.)
"""
import json

import pytest

_EXAMPLE = "geom_conico_chico.json"   # fastest shipped geometry, film-less


@pytest.fixture(scope="session", params=["film_less", "film_tip"])
def mesh_result(request, repo_root, tmp_path_factory):
    """Build the mesh once per configuration (geom -> Params -> mallar, exactly
    like launch_mallador.py's batch mode), and precompute the expected contract."""
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")

    import capa1_modelo
    import capa4_malla

    case = request.param
    cfg = json.loads(
        (repo_root / "1_mesher" / "examples" / _EXAMPLE).read_text(encoding="utf-8")
    )
    params = dict(cfg["params"])
    if case == "film_tip":                       # derive a with-film variant
        params["include_film_tip"] = True
        params["delta_film_tip"] = 10e-9
    p = capa1_modelo.Params(**params)

    out_dir = tmp_path_factory.mktemp(f"mesh_{case}")
    name = f"conico_chico_{case}"
    info = capa4_malla.mallar(p, str(out_dir), name)
    limits = json.loads((out_dir / f"{name}_limits.json").read_text(encoding="utf-8"))

    # Expected contract for this configuration (which tags MUST be declared, and
    # which subdomain tags the domain mesh may carry).
    TAGS = capa1_modelo.TAGS
    expected_tags = {
        "AXIS": TAGS["AXIS"], "WALL": TAGS["WALL"],
        "INLET": TAGS["INLET"], "OUTLET": TAGS["OUTLET"],
        "CHARGE_ZONE_TIP": TAGS["CHARGE_ZONE_TIP"],
        "CHARGE_ZONE_BASE": TAGS["CHARGE_ZONE_BASE"],
        "DOMAIN_FLUID": TAGS["DOMAIN_FLUID"],
    }
    expected_domain = {TAGS["DOMAIN_FLUID"]}
    if p.include_film_tip:
        expected_tags["FILM_TIP_INTERFACE"] = TAGS["FILM_TIP_INTERFACE"]
        expected_tags["DOMAIN_FILM_TIP"] = TAGS["DOMAIN_FILM_TIP"]
        expected_domain.add(TAGS["DOMAIN_FILM_TIP"])
    if p.include_film_base:
        expected_tags["FILM_BASE_INTERFACE"] = TAGS["FILM_BASE_INTERFACE"]
        expected_tags["DOMAIN_FILM_BASE"] = TAGS["DOMAIN_FILM_BASE"]
        expected_domain.add(TAGS["DOMAIN_FILM_BASE"])

    return {
        "case": case, "dir": out_dir, "name": name, "info": info, "limits": limits,
        "p": p, "TAGS": TAGS,
        "expected_tags": expected_tags, "expected_domain": expected_domain,
    }


def test_mesh_has_cells(mesh_result):
    """The mesher reports a non-empty mesh."""
    info = mesh_result["info"]
    assert info["n_tri"] > 0, "mesh has no triangles"
    assert info["n_lin"] > 0, "mesh has no boundary segments"


def test_contract_files_exist(mesh_result):
    """The files the Solver reads are written and non-empty."""
    out, name = mesh_result["dir"], mesh_result["name"]
    for suffix in ("_domain.xdmf", "_domain.h5", "_facets.xdmf", "_facets.h5", "_limits.json"):
        f = out / f"{name}{suffix}"
        assert f.is_file() and f.stat().st_size > 0, f"missing or empty: {f.name}"


def test_limits_json_shape(mesh_result):
    """<name>_limits.json echoes the geometry with a sane, consistent shape."""
    lim, name, p = mesh_result["limits"], mesh_result["name"], mesh_result["p"]
    assert lim["m_name"] == name
    assert lim["channel_type"] == "conical"
    for key in ("z_tip", "z_base", "R_tip", "R_base", "L_pore", "L_res", "R_res"):
        v = lim[key]
        assert isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0, \
            f"{key} is not a positive number"
    assert lim["z_base"] > lim["z_tip"], "z_base must lie downstream of z_tip"
    assert lim["include_film_tip"] == p.include_film_tip
    assert lim["include_film_base"] == p.include_film_base
    # When a tip film is present, its geometry fields are echoed for the Solver.
    if p.include_film_tip:
        assert lim["delta_film_tip"] > 0, "delta_film_tip missing/invalid"
        assert "z_film_tip" in lim, "z_film_tip not echoed"


def test_tags_match_contract(mesh_result):
    """limits.json declares exactly the tags the configuration requires, with
    the contract's integer values."""
    assert mesh_result["limits"]["tags"] == mesh_result["expected_tags"]


def test_exported_meshes_carry_contract_tags(mesh_result):
    """Re-read the XDMF meshes: subdomain tags match the configuration, facet
    tags stay within the contract, and the essential boundaries are present."""
    meshio = pytest.importorskip("meshio")
    out, name, TAGS = mesh_result["dir"], mesh_result["name"], mesh_result["TAGS"]

    dom = meshio.read(str(out / f"{name}_domain.xdmf"))
    tri_tags = {int(t) for arr in dom.cell_data["subdomains"] for t in arr}
    assert tri_tags == mesh_result["expected_domain"], f"unexpected domain tags: {tri_tags}"

    fac = meshio.read(str(out / f"{name}_facets.xdmf"))
    facet_tags = {int(t) for arr in fac.cell_data["f"] for t in arr}
    assert facet_tags <= set(TAGS.values()), \
        f"facet tags outside contract: {facet_tags - set(TAGS.values())}"
    for essential in ("AXIS", "WALL", "INLET", "OUTLET"):
        assert TAGS[essential] in facet_tags, f"missing boundary: {essential}"
    if mesh_result["p"].include_film_tip:
        assert TAGS["FILM_TIP_INTERFACE"] in facet_tags, "missing FILM_TIP_INTERFACE facet"


def test_mesh_geometry_invariants(mesh_result):
    """Geometric sanity of the exported domain mesh: coordinates are finite and
    within the expected axisymmetric bounding box, the axis (r=0) is present, no
    node has negative radius, and no triangle is degenerate. Unlike the tag
    tests, this checks the SHAPE of the mesh, not just its labels."""
    np = pytest.importorskip("numpy")
    meshio = pytest.importorskip("meshio")
    out, name, lim = mesh_result["dir"], mesh_result["name"], mesh_result["limits"]

    dom = meshio.read(str(out / f"{name}_domain.xdmf"))
    pts = dom.points
    assert np.isfinite(pts).all(), "mesh has non-finite coordinates (NaN/inf)"
    z, r = pts[:, 0], pts[:, 1]

    # Expected axisymmetric bounding box, from the geometry the mesher echoed
    # into limits.json (so this stays valid for both configurations).
    z_outlet = lim["z_base"] + lim["L_res"]      # = 2*L_res + L_pore
    r_max = lim["R_res"]
    atol = 1e-6 * z_outlet                        # tolerance tied to domain scale

    # z runs from the inlet (0) to the outlet; r from the axis (0) to the reservoir.
    assert abs(z.min()) <= atol, f"z does not start at the inlet (z_min={z.min():.3e})"
    assert np.isclose(z.max(), z_outlet, rtol=1e-4, atol=atol), \
        f"z_max={z.max():.3e} != expected outlet {z_outlet:.3e}"
    assert abs(r.min()) <= atol, f"the axis (r=0) is missing (r_min={r.min():.3e})"
    assert np.isclose(r.max(), r_max, rtol=1e-4, atol=atol), \
        f"r_max={r.max():.3e} != reservoir radius {r_max:.3e}"

    # Axisymmetric domain: no node may have negative radius.
    assert (r >= -atol).all(), "found nodes with negative radius"

    # No degenerate/collapsed triangle (tiny EDL-refined elements are legitimate).
    tris = dom.get_cells_type("triangle")
    z0, r0 = z[tris[:, 0]], r[tris[:, 0]]
    z1, r1 = z[tris[:, 1]], r[tris[:, 1]]
    z2, r2 = z[tris[:, 2]], r[tris[:, 2]]
    area = 0.5 * np.abs((z1 - z0) * (r2 - r0) - (z2 - z0) * (r1 - r0))
    assert (area > 0.0).all(), "mesh contains degenerate (zero-area) triangles"

    # The meshed fluid area is positive and fits inside the bounding box.
    bbox = (z.max() - z.min()) * (r.max() - r.min())
    assert 0.0 < area.sum() < bbox, "total mesh area is implausible"
