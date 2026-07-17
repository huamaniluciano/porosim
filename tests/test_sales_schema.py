# -*- coding: utf-8 -*-
"""
T1 - Schema test: 2_solver/sales.json (salt catalog).

sales.json is the SINGLE source of truth for each electrolyte's properties
(symbol, charge z, diffusivity D, solubility, Ksp). The solver loads it and
exports the chosen salt's block into the _sim.json. An error here -- an anion
with positive z, a negative D, a "soluble" salt carrying a Ksp -- would
propagate into EVERY simulation. This test locks that contract down.

The suite is parametrized per salt (pytest_generate_tests) so that, on
failure, the report says EXACTLY which salt and which field.

(The JSON keys -- salts, cation, anion, soluble, Ksp_M2, D_m2s, ... -- are
part of POROSIM's contract and used verbatim.)
"""
import json
import pathlib


def _load_sales(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# At collection time there are no fixtures yet; resolve the path by hand.
_SALES_PATH = pathlib.Path(__file__).resolve().parent.parent / "2_solver" / "sales.json"


def pytest_generate_tests(metafunc):
    """Generate one test case per salt in the catalog."""
    if "salt_id" in metafunc.fixturenames:
        ids = list(_load_sales(_SALES_PATH)["salts"].keys())
        metafunc.parametrize("salt_id", ids)


def test_schema_top_level(sales_json_path):
    """The file has a schema version and at least one salt."""
    data = _load_sales(sales_json_path)
    assert data.get("schema_version") == 1
    salts = data.get("salts")
    assert isinstance(salts, dict) and salts, "there must be at least one salt"


def test_salt_entry(sales_json_path, salt_id):
    """Each salt entry satisfies the sales.json contract."""
    salt = _load_sales(sales_json_path)["salts"][salt_id]

    # The dict key and the 'name' field must match.
    assert salt.get("name") == salt_id, f"{salt_id}: 'name' does not match the key"

    # Ions present and well-formed; the cation has z>0 and the anion z<0.
    for role, must_be_positive in (("cation", True), ("anion", False)):
        ion = salt.get(role)
        assert isinstance(ion, dict), f"{salt_id}: missing '{role}'"

        symbol = ion.get("symbol")
        assert isinstance(symbol, str) and symbol.strip(), \
            f"{salt_id}.{role}.symbol is empty or invalid"

        z = ion.get("z")
        assert isinstance(z, int) and not isinstance(z, bool) and z != 0, \
            f"{salt_id}.{role}.z must be a non-zero integer"
        assert (z > 0) == must_be_positive, \
            f"{salt_id}.{role}: wrong sign of z (z={z})"

        D = ion.get("D_m2s")
        assert isinstance(D, (int, float)) and not isinstance(D, bool) and D > 0, \
            f"{salt_id}.{role}.D_m2s must be > 0"

    # Solubility <-> Ksp:  soluble => Ksp null ;  insoluble => Ksp > 0.
    soluble = salt.get("soluble")
    assert isinstance(soluble, bool), f"{salt_id}.soluble must be a bool"
    ksp = salt.get("Ksp_M2")
    if soluble:
        assert ksp is None, f"{salt_id}: a soluble salt must not declare Ksp_M2"
    else:
        assert isinstance(ksp, (int, float)) and not isinstance(ksp, bool) and ksp > 0, \
            f"{salt_id}: an insoluble salt needs Ksp_M2 > 0"
