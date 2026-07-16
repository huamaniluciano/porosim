# -*- coding: utf-8 -*-
"""
Módulo: Mapa de Potential 2D
Categoría: potential_map

Genera un mapa de color 2D del potencial eléctrico phi para un voltaje
elegido por el usuario, usando tricontourf (igual que ion_maps).

Funciona con o sin films. Cada film se dibuja como un borde:
    rojo  si el film tiene carga fija positiva
    azul  si el film tiene carga fija negativa
El interior del film NO se rellena (ahí va la info del potencial).

API pura (compartida entre consola, batch y GUI):
    crear_figura(datos, ctx, con_campo=False, ...)   → matplotlib Figure

Contrato del extractor (ver modulos/MODULE_CONTRACT.md):
    preparar(ruta, v_label=None, sol=None)  → (datos, ctx) | None
    guardar(datos, ctx, ruta, png, con_datos) → [Paths]
    procesar(ruta)                           → cáscara de consola

datos = {"z_nm", "r_nm", "tri", "phi_V"} (arrays SIN espejar, φ en Volts);
ctx   = porosim_comun.contexto_de(meta, stem, v_label).

con_campo=True superpone las líneas de campo E = -∇φ (ver campo_electrico()
en porosim_comun-nivel; acá se calcula con el mismo CubicTriInterpolator).

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc


# =============================================================================
# CAMPO ELÉCTRICO E = -∇φ (pura NumPy/matplotlib.tri) — compartido con field_lines
# =============================================================================
def campo_electrico(z_nm, r_nm, tri, phi_V):
    """E en grilla regular + espejo axisimétrico (Ez par, Er impar), con
    CubicTriInterpolator directo sobre la malla (sin project()). Devuelve
    {Z, R_full, Ez, Er} listo para streamplot (NaN fuera del dominio)."""
    triang = mtri.Triangulation(z_nm, r_nm, tri)

    z_min, z_max = z_nm.min(), z_nm.max()
    r_min, r_max = r_nm.min(), r_nm.max()
    aspect = (z_max - z_min) / (r_max - r_min) if (r_max - r_min) > 0 else 1.0
    n_r = 55
    n_z = int(np.clip(n_r * aspect, 80, 400))
    Z = np.linspace(z_min, z_max, n_z)
    R = np.linspace(r_min, r_max, n_r)
    Z_grid, R_grid = np.meshgrid(Z, R)

    interp = mtri.CubicTriInterpolator(triang, phi_V, kind="geom")
    dphi_dz, dphi_dr = interp.gradient(Z_grid, R_grid)
    Ez = np.ma.filled(-dphi_dz, np.nan)   # NaN = fuera del dominio (streamplot)
    Er = np.ma.filled(-dphi_dr, np.nan)

    R_neg, Ez_neg, Er_neg = -R[::-1], Ez[::-1, :], -Er[::-1, :]
    if np.isclose(R_neg[-1], R[0]):       # evitar fila duplicada en r = 0
        R_neg, Ez_neg, Er_neg = R_neg[:-1], Ez_neg[:-1, :], Er_neg[:-1, :]
    return {"Z": Z, "R_full": np.concatenate([R_neg, R]),
            "Ez": np.vstack([Ez_neg, Ez]), "Er": np.vstack([Er_neg, Er])}


# =============================================================================
# FIGURA (la misma para consola y GUI). con_campo → líneas de campo encima.
# =============================================================================
def crear_figura(datos, ctx, con_campo=False, leyenda=True,
                 ylim=None, xlim=None, titulo=True, figsize=None,
                 campo=None):
    """campo: dict {Z, R_full, Ez, Er} ya calculado (p.ej. cacheado por la
    GUI). Si es None y con_campo=True, se calcula acá con campo_electrico()."""
    z_nm, r_nm, tri, phi_V = (datos["z_nm"], datos["r_nm"],
                              datos["tri"], datos["phi_V"])
    z2, r2, tri2, phi2 = pc.espejo(z_nm, r_nm, tri, phi_V)
    norm, niveles = pc.escala_potencial(phi_V, ctx.get("V_T"))

    ec = campo
    if con_campo and ec is None:
        ec = campo_electrico(z_nm, r_nm, tri, phi_V)

    if figsize is None:
        figsize = (14, 6) if con_campo else (14, 5)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.tricontourf(z2, r2, tri2, phi2, levels=niveles,
                        cmap="RdBu_r", norm=norm, extend="both")
    if ec is not None:
        ax.streamplot(ec["Z"], ec["R_full"], ec["Ez"], ec["Er"],
                      color="0.55", density=1.4, linewidth=0.8,
                      arrowsize=1.0, arrowstyle="->")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Potential [V]")
    pc.marcar_colorbar(cbar)

    ax.set_aspect("equal")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_facecolor("0.93")   # fuera del dominio (membrana) = gris clarito
    if titulo:
        base = "Electric Field Lines" if con_campo else "2D Electric Potential Map"
        ax.set_title(f"{base} — {ctx['v_label']}V  |  {ctx['stem']}")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")

    pc.guias_canal(ax, ctx)
    pc.rectangulos_films(ax, ctx, r_max_nm=r2.max(), con_labels=True)
    if leyenda and ctx["films"]:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None,
             titulo="2D ELECTRIC POTENTIAL MAP"):
    """→ (datos, ctx) o None. datos = {z_nm, r_nm, tri, phi_V}.
    v_label=None → pregunta por consola; con valor → valida (batch).
    sol = pc.cargar_solucion() ya cargado (batch multi-voltaje) o None.
    `titulo` permite a field_lines reusar este preparar con su banner."""
    prep = pc.preparar_comun(ruta_solucion, titulo, v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    dm    = sol["dm"]
    phi_V = campos["phi_adim"] * ctx["V_T"]
    print(f"    φ en [{phi_V.min():.4f} V, {phi_V.max():.4f} V]")
    datos = {"z_nm": dm["z_nm"], "r_nm": dm["r_nm"], "tri": dm["tri"],
             "phi_V": phi_V}
    return datos, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False,
            con_campo=False, sufijo="potential_map"):
    """Mapa limpio publication-ready (mismo output que el '¿Guardar mapa
    limpio?' de consola): {stem}_{sufijo}_{V}V.png. Sin datos tabulares
    (con_datos se acepta por uniformidad del contrato y se ignora).
    `con_campo`/`sufijo` permiten a field_lines reusar este guardar."""
    rutas = []
    if png:
        figc = crear_figura(datos, ctx, con_campo=con_campo, leyenda=False,
                            titulo=False, figsize=(10, 4))
        stem = pathlib.Path(ruta_solucion).stem
        rutas.append(pc.guardar_figura(
            figc, ruta_solucion, f"{stem}_{sufijo}_{ctx['v_label']}V.png",
            limpio=True))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep

    print(">>> Generating 2D map...")
    crear_figura(datos, ctx, con_campo=False)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()

    resp = input("\nSave clean map (content + scale only, "
                 "no axes/legends)? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion)
    else:
        print("    (no image saved)")
