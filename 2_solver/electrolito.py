# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — SOLVER · ELECTROLITO (v1.1.0)
  Electrolyte → list of ions that the PNP engine solves for. No FEniCS.
═══════════════════════════════════════════════════════════════════════════

The engine solves U = (φ, u_1, ..., u_N) with u_i = ln(c_i / c0_i): one
Nernst-Planck equation per ion, and all of them in the charge density of
Poisson. This module only decides WHICH ions and at what bulk concentration;
it imports nothing heavy, so the batch validation and the tests use it without
FEniCS.

An ion is a dict {"symbol", "z", "D_m2s", "c0_mM"} (c0_mM = bulk concentration
of THAT ion). The order of the list is the order of the u_i components of U.

The engine (motor_pnp.resolver) receives the electrolyte in one of two forms:

  · binary, as in the paper:  cfg["salt"] (sales.json schema) + cfg["c0_mM"]
  · mixture of N ions:        cfg["ions"] = [ion, ...] (+ optional
                              cfg["electrolyte"] / cfg["electrolyte_label"])

A SYMMETRIC salt (z+ = −z−, e.g. KCl, MgSO4) runs in binary mode: every
expression, file name, checkpoint key and _sim.json field is the one of
v1.0.0. An ASYMMETRIC salt (e.g. CaCl2) given as "salt" + "c0_mM" runs in
general mode with its neutral stoichiometry (Ca²⁺ c0, Cl⁻ 2·c0): v1.0.0 put
c0 on both ions, so the bulk of an asymmetric salt was not electroneutral.
"""
import math
import unicodedata

import numpy as np


# Tolerance of the bulk electroneutrality check, relative to Σ|z|·c0.
TOL_NEUTRALIDAD = 1e-9


def es_simetrica(sal):
    """True if the salt is z:z (z+ = −z−): binary mode of the paper."""
    return int(sal["cation"]["z"]) == -int(sal["anion"]["z"])


def iones_de_sal(sal, c0_mM):
    """
    Ions of ONE salt at salt concentration c0_mM, with its neutral
    stoichiometry: ν+ = |z−|/g and ν− = z+/g, g = gcd(z+, |z−|).
    KCl 100 → K⁺ 100, Cl⁻ 100 · CaCl2 10 → Ca²⁺ 10, Cl⁻ 20 · MgSO4 → 1:1.

    For a symmetric salt ν+ = ν− = 1 and c0_mM is passed through untouched
    (same type and value as v1.0.0 used it).
    """
    for rol in ("cation", "anion"):
        z = sal[rol]["z"]
        if isinstance(z, bool) or not float(z).is_integer():
            raise ValueError(f'Salt "{sal.get("name", "?")}": the {rol} valence must '
                             f'be an integer (got z={z!r}).')
    z_p, z_m = int(sal["cation"]["z"]), int(sal["anion"]["z"])
    if z_p <= 0 or z_m >= 0:
        raise ValueError(f'Salt "{sal.get("name", "?")}": the cation needs z > 0 '
                         f'and the anion z < 0 (got z+={z_p}, z-={z_m}).')
    g = math.gcd(z_p, -z_m)
    nu_p, nu_m = (-z_m) // g, z_p // g
    return [{"symbol": sal["cation"].get("symbol", "cation"),
             "z":      sal["cation"]["z"],
             "D_m2s":  sal["cation"]["D_m2s"],
             "c0_mM":  c0_mM * nu_p if nu_p != 1 else c0_mM},
            {"symbol": sal["anion"].get("symbol", "anion"),
             "z":      sal["anion"]["z"],
             "D_m2s":  sal["anion"]["D_m2s"],
             "c0_mM":  c0_mM * nu_m if nu_m != 1 else c0_mM}]


def combinar_sales(entradas):
    """
    Mixture of salts → ions. `entradas` = [(salt_dict, c0_mM), ...].

    Each salt contributes its ions with its neutral stoichiometry; an ion that
    appears in several salts (same symbol, e.g. the Cl⁻ of KCl and of HCl) is
    SUMMED, and must carry the same z and D in all of them.

    Returns (ions, info, label):
      ions   [{"symbol", "z", "D_m2s", "c0_mM"}] in order of appearance
      info   [{"salt": name, "c0_mM": c}]   (goes to the _sim.json)
      label  "KCl_100mM+HCl_10mM"            (file and folder names)
    """
    if not entradas:
        raise ValueError("The electrolyte needs at least one salt.")
    iones, info, partes = [], [], []
    for sal, c in entradas:
        c = float(c)
        if not c > 0:
            raise ValueError(f'Salt "{sal.get("name", "?")}": c0_mM must be > 0 (got {c}).')
        for ion in iones_de_sal(sal, c):
            previo = next((i for i in iones if i["symbol"] == ion["symbol"]), None)
            if previo is None:
                iones.append({"symbol": ion["symbol"], "z": int(ion["z"]),
                              "D_m2s": float(ion["D_m2s"]), "c0_mM": float(ion["c0_mM"])})
                continue
            if previo["z"] != int(ion["z"]) or previo["D_m2s"] != float(ion["D_m2s"]):
                raise ValueError(
                    f'Ion "{ion["symbol"]}" appears in two salts with a different z '
                    f'or D: ({previo["z"]}, {previo["D_m2s"]}) vs ({ion["z"]}, '
                    f'{ion["D_m2s"]}). Unify them in sales.json.')
            previo["c0_mM"] += float(ion["c0_mM"])
        info.append({"salt": sal.get("name", "custom"), "c0_mM": c})
        partes.append(f"{sal.get('name', 'custom')}_{c:g}mM")
    validar_iones(iones)
    return iones, info, "+".join(partes)


def validar_iones(iones):
    """Checks the engine needs before building U (raises ValueError):
    ≥ 2 ions, distinct symbols, integer z ≠ 0, D > 0, c0 > 0 (u = ln(c/c0))
    and an electroneutral bulk (Σ z·c0 = 0; otherwise c_i = c0_i on the
    reservoirs is a charged boundary condition)."""
    if not isinstance(iones, (list, tuple)) or len(iones) < 2:
        raise ValueError("At least 2 ions are needed (a cation and an anion).")
    simbolos = [str(i.get("symbol", "")) for i in iones]
    if len(set(simbolos)) != len(simbolos) or not all(simbolos):
        raise ValueError(f"Each ion needs its own non-empty symbol: {simbolos}")
    for ion in iones:
        z = ion["z"]
        if isinstance(z, bool) or not float(z).is_integer() or z == 0:
            raise ValueError(f'Ion "{ion["symbol"]}": z must be a non-zero integer (got {z!r}).')
        if not float(ion["D_m2s"]) > 0:
            raise ValueError(f'Ion "{ion["symbol"]}": D_m2s must be > 0.')
        if not float(ion["c0_mM"]) > 0:
            raise ValueError(f'Ion "{ion["symbol"]}": c0_mM must be > 0 '
                             f'(log variables u = ln(c/c0)).')
    carga  = sum(i["z"] * float(i["c0_mM"]) for i in iones)
    escala = sum(abs(i["z"]) * float(i["c0_mM"]) for i in iones)
    if abs(carga) > TOL_NEUTRALIDAD * escala:
        raise ValueError("The bulk is not electroneutral: sum(z*c0) = "
                         f"{carga:+.6g} mM (ions: "
                         f"{[(i['symbol'], i['z'], i['c0_mM']) for i in iones]}).")


def _ascii(texto):
    """'ClO₄⁻' → 'ClO4' (only letters and digits, for file names)."""
    plano = unicodedata.normalize("NFKD", str(texto))
    return "".join(ch for ch in plano if ch.isascii() and ch.isalnum())


def etiqueta_de_iones(iones):
    """Default label of a hand-built "ions" list: 'K100+Cl110+H10mM'."""
    return "+".join(f"{_ascii(i['symbol']) or 'ion'}{float(i['c0_mM']):g}"
                    for i in iones) + "mM"


def iones_de_cfg(cfg):
    """
    (ions, general_mode, label) of a resolver() cfg.

      cfg["ions"]               → mixture: general_mode = True,
                                  label = cfg["electrolyte_label"] or a default
      cfg["salt"] symmetric     → binary mode of the paper (general_mode = False),
                                  label = "<salt>_<c0_mM>mM" (v1.0.0 file names)
      cfg["salt"] asymmetric    → general mode with the neutral stoichiometry,
                                  same label as the binary mode
    """
    if cfg.get("ions"):
        iones = [{"symbol": str(ion.get("symbol", f"ion{k}")),
                  "z":      ion["z"],
                  "D_m2s":  float(ion["D_m2s"]),
                  "c0_mM":  float(ion["c0_mM"])}
                 for k, ion in enumerate(cfg["ions"])]
        validar_iones(iones)            # before int(): z = 1.5 must not pass as 1
        for ion in iones:
            ion["z"] = int(ion["z"])
        return iones, True, (cfg.get("electrolyte_label") or etiqueta_de_iones(iones))

    sal, c0_mM = cfg["salt"], cfg["c0_mM"]
    iones = iones_de_sal(sal, c0_mM)
    validar_iones(iones)
    return iones, (not es_simetrica(sal)), f"{sal['name']}_{c0_mM}mM"


def donnan_psi(iones, c_fix):
    """
    Dimensionless Donnan potential ψ = F·φ_D/(R·T) of a film with fixed charge
    c_fix [mol/m³ = mM] in equilibrium with a bulk of N ions:

        Σ z_i·c0_i·exp(−z_i·ψ) + c_fix = 0     (electroneutrality in the film)

    g(ψ) is strictly decreasing (g' = −Σ z_i²·c_i < 0), so the root is unique:
    bisection on a bracket that doubles until it holds the root. For a 1:1 salt
    it returns arcsinh(c_fix / 2c0).
    """
    def g(psi):
        return sum(ion["z"] * ion["c0_mM"] * np.exp(-ion["z"] * psi)
                   for ion in iones) + c_fix

    lo, hi = -1.0, 1.0
    while g(lo) < 0:
        lo *= 2.0
    while g(hi) > 0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
