# -*- coding: utf-8 -*-
"""
Módulo: Mapa de Concentration de Total Ions y Ionic Strength 2D
Categoría: ion_maps

Genera mapas de color 2D side-by-side de la concentración total de iones
(cation + anion) y la fuerza iónica del medio. Lee c0, valencias y nombres de
iones desde el _sim.json. Visualiza la región del film si existe.

API pura (compartida entre procesar() de consola y la GUI):
    crear_figura(datos, ctx, con_fuerza_ionica=True, ...)   → matplotlib Figure

datos = {"z_nm", "r_nm", "tri", "cp_mM", "cm_mM"} (arrays SIN espejar);
ctx   = porosim_comun.contexto_de(meta, stem, v_label).

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

_AQUI = pathlib.Path(__file__).resolve().parent
_MOD  = str(_AQUI.parent)
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))
import porosim_comun as pc
import ions               # mismo directorio (ion_maps/): reusa preparar()


def _rectangulos_desde_eje(ax, ctx, r_max_nm):
    """Films como rectángulo desde el eje (0) hasta r_max: este módulo NO
    espeja (grafica solo r ≥ 0), así que el rectángulo arranca en r = 0."""
    pc.rectangulos_films(ax, ctx, r_max_nm=r_max_nm, r_min_nm=0.0, con_labels=True)


def crear_figura(datos, ctx, con_fuerza_ionica=True, leyenda=True,
                 titulo=True, figsize=None):
    """Iones totales (siempre) + fuerza iónica (opcional). Sin espejo: grafica
    el semiplano r ≥ 0 de la malla (como el módulo original)."""
    sal = ctx["salt"]
    z_nm, r_nm, tri = datos["z_nm"], datos["r_nm"], datos["tri"]
    ct = datos["cp_mM"] + datos["cm_mM"]
    fi = 0.5 * (datos["cp_mM"] * sal["z_p"]**2 + datos["cm_mM"] * sal["z_m"]**2)
    c0, r_max = ctx["c0"], r_nm.max()

    niveles_tot = np.linspace(1.20 * c0, 5.00 * c0, 100)

    if figsize is None:
        figsize = (16, 5) if con_fuerza_ionica else (9, 5)
    if con_fuerza_ionica:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        cab = "Total Ions y Ionic Strength"
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
        cab = "Total Ions"
    if titulo:
        fig.suptitle(f"{cab} — {ctx['v_label']}V  |  {ctx['stem']}", fontsize=12)

    im1 = ax1.tricontourf(z_nm, r_nm, tri, ct, levels=niveles_tot,
                          cmap="jet", extend="both")
    ax1.set_title(f"Total Ions ({sal['label_p']} + {sal['label_m']})")
    ax1.set_xlabel("z [nm]")
    ax1.set_ylabel("r [nm]")
    ax1.set_aspect("equal")
    pc.marcar_colorbar(fig.colorbar(im1, ax=ax1, label="Concentration [mM]"))

    if con_fuerza_ionica:
        factor = 0.5 * (sal["z_p"]**2 + sal["z_m"]**2)
        niveles_fi = np.linspace(0.60 * c0 * factor, 2.50 * c0 * factor, 100)
        im2 = ax2.tricontourf(z_nm, r_nm, tri, fi, levels=niveles_fi,
                              cmap="jet", extend="both")
        ax2.set_title("Ionic Strength ($I$)")
        ax2.set_xlabel("z [nm]")
        ax2.set_aspect("equal")
        pc.marcar_colorbar(fig.colorbar(im2, ax=ax2, label="Ionic Strength [mM]"))

    for ax in [a for a in (ax1, ax2) if a is not None]:
        pc.guias_canal(ax, ctx)
        _rectangulos_desde_eje(ax, ctx, r_max)
        if leyenda and ctx["films"]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """Mismos datos que ions.py (cp/cm nodales); solo cambia el banner."""
    return ions.preparar(ruta_solucion, v_label, sol,
                          titulo="MAPA DE IONES TOTALES Y FUERZA IÓNICA 2D")


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False,
            con_fuerza_ionica=True):
    """{stem}_total_ions_{V}V.png. Se guarda ANOTADO (con ejes y títulos
    de panel): al ser 2 paneles, la versión "limpia" los volvería
    indistinguibles. Sin datos tabulares (con_datos se ignora)."""
    rutas = []
    if png:
        fig  = crear_figura(datos, ctx, con_fuerza_ionica=con_fuerza_ionica)
        stem = pathlib.Path(ruta_solucion).stem
        rutas.append(pc.guardar_figura(
            fig, ruta_solucion, f"{stem}_total_ions_{ctx['v_label']}V.png"))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep

    ct = datos["cp_mM"] + datos["cm_mM"]
    fi = 0.5 * (datos["cp_mM"] * ctx["salt"]["z_p"]**2
                + datos["cm_mM"] * ctx["salt"]["z_m"]**2)
    print(f"    Total Ions: [{ct.min():.2f}, {ct.max():.2f}] mM")
    print(f"    Ionic Strength: [{fi.min():.2f}, {fi.max():.2f}] mM")

    resp = input("\nInclude the Ionic Strength panel? "
                 "[Enter=Yes / n=Total Ions only]: ").strip().lower()
    con_fi = resp not in ("n", "no")

    print(">>> Generating 2D maps...")
    crear_figura(datos, ctx, con_fuerza_ionica=con_fi)
    print(">>> Opening interactive window. Close it to return to the menu.")
    plt.show()

    resp = input("\nSave the map (PNG)? [Enter=No / y=Yes]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion, con_fuerza_ionica=con_fi)
    else:
        print("    (no image saved)")
