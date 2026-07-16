# -*- coding: utf-8 -*-
"""
Módulo: Perfil de Concentration Promedio en el Canal
Categoría: axial_profiles

Para cada posición z dentro del canal, calcula la concentración promedio sobre
la sección transversal usando integración radial ponderada por área:

    <c>(z) = ∫₀^R(z) c(z,r) · r · dr  /  ∫₀^R(z) r · dr
           = 2/R(z)² · ∫₀^R(z) c(z,r) · r · dr

R(z) se obtiene de los METADATOS del cono (R_tip, R_base, z_tip, z_base): es
lineal entre las dos bocas. No necesita el _facets.xdmf.

⚠ SUPONE canal CÓNICO (R lineal entre bocas). Para geometrías no lineales
(bullet, spline curvo) habría que rastrear la pared real (tag WALL).

Limitación conocida: falla si hay objetos sólidos dentro del canal (la sección
transversal dejaría de ser un disco completo).

API pura (compartida entre procesar() de consola y la GUI):
    perfil_promedio_seccion(z_nm, r_nm, tri, cp, cm, z_tip, z_base,
                            R_tip, R_base, c0)        → {z_nm, cp_prom, cm_prom, ct_prom} | None
    crear_figura(datos_prom, ctx, curvas, ...)        → Figure de 1 panel

El promedio se hace por interpolación triangular LINEAL sobre los arrays de la
malla (NumPy puro): idéntico camino para consola y GUI.

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

# Importa la primitiva de trazado del perfil axial para no re-implementar el
# mismo estilo de curvas/bulk/films (misma familia de figura).
_PERF = str(pathlib.Path(__file__).resolve().parent)
if _PERF not in sys.path:
    sys.path.insert(0, _PERF)
import axis_profile_ions as pzi

CURVAS_DEFAULT = pzi.CURVAS_DEFAULT


# =============================================================================
# CÁLCULO (puro NumPy: interpolación triangular lineal)
# =============================================================================
def perfil_promedio_seccion(z_nm, r_nm, tri, cp_mM, cm_mM,
                            z_tip, z_base, R_tip, R_base, c0,
                            N_Z=200, N_R=35):
    """<c>(z) = 2/R(z)² ∫ c(z,r)·r dr por interpolación triangular lineal.
    Devuelve None si faltan los metadatos de geometría del cono."""
    if None in (z_tip, z_base, R_tip, R_base):
        return None
    triang    = mtri.Triangulation(z_nm, r_nm, tri)
    interp_cp = mtri.LinearTriInterpolator(triang, cp_mM)
    interp_cm = mtri.LinearTriInterpolator(triang, cm_mM)

    z_canal   = np.linspace(z_tip, z_base, N_Z)
    z_grid_nm = z_canal * 1e9
    cp_prom   = np.zeros(N_Z)
    cm_prom   = np.zeros(N_Z)

    for i, z in enumerate(z_canal):
        t   = (z - z_tip) / (z_base - z_tip) if (z_base != z_tip) else 0.0
        R_z = R_tip + (R_base - R_tip) * t
        r_pts_nm = np.linspace(1e-3, R_z * 0.999 * 1e9, N_R)
        z_pts_nm = np.full(N_R, z_grid_nm[i])

        cp_r = np.nan_to_num(interp_cp(z_pts_nm, r_pts_nm), nan=c0)
        cm_r = np.nan_to_num(interp_cm(z_pts_nm, r_pts_nm), nan=c0)

        R_nm = R_z * 1e9
        if R_nm > 0:
            cp_prom[i] = 2.0 / (R_nm**2) * np.trapz(cp_r * r_pts_nm, r_pts_nm)
            cm_prom[i] = 2.0 / (R_nm**2) * np.trapz(cm_r * r_pts_nm, r_pts_nm)
        else:
            cp_prom[i], cm_prom[i] = c0, c0

    return {"z_nm": z_grid_nm, "cp_prom": cp_prom, "cm_prom": cm_prom,
            "ct_prom": cp_prom + cm_prom}


# =============================================================================
# FIGURA (misma familia que el perfil axial, pero con datos promediados)
# =============================================================================
def crear_figura(datos_prom, ctx, curvas=CURVAS_DEFAULT, leyenda=True,
                 titulo=True, figsize=(11, 5)):
    """Reusa el trazado del perfil axial pero con <c>(z). Etiquetas <c_·>."""
    datos_axial = {"z_axis_nm": datos_prom["z_nm"],
                   "cp_ax_mM":  datos_prom["cp_prom"],
                   "cm_ax_mM":  datos_prom["cm_prom"],
                   "ct_ax_mM":  datos_prom["ct_prom"]}
    fig, ax = plt.subplots(figsize=figsize)
    if titulo:
        fig.suptitle(f"Section-Average Concentration <c>(z) — {ctx['v_label']}V  |  "
                     f"{ctx['stem']}", fontsize=12)
    pzi.dibujar_iones_axial(ax, datos_axial, ctx, curvas=curvas, leyenda=leyenda)
    ax.set_ylabel("Average concentration [mM]")
    fig.tight_layout()
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """→ (datos_prom, ctx) o None.
    datos_prom = {z_nm, cp_prom, cm_prom, ct_prom} (promedio en sección)."""
    prep = pc.preparar_comun(ruta_solucion, "CHANNEL AVERAGE PROFILE",
                             v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep

    if None in (ctx["z_tip"], ctx["z_base"], ctx["R_tip"], ctx["R_base"]):
        print("❌ Missing z_tip/z_base/R_tip/R_base in the JSON.")
        print("   They are needed to build R(z) of the cone.")
        return None
    print(f"  R(z): [{ctx['R_tip']*1e9:.1f}, {ctx['R_base']*1e9:.1f}] nm")

    dm = sol["dm"]
    cp = ctx["c0"] * np.exp(campos["up"])
    cm = ctx["c0"] * np.exp(campos["um"])

    print(">>> Integrating cross-sections (triangular interpolation)...")
    datos_prom = perfil_promedio_seccion(
        dm["z_nm"], dm["r_nm"], dm["tri"], cp, cm,
        ctx["z_tip"], ctx["z_base"], ctx["R_tip"], ctx["R_base"], ctx["c0"])
    if datos_prom is None:
        print("❌ Could not compute the average (missing cone metadata).")
        return None
    return datos_prom, ctx


def guardar(datos_prom, ctx, ruta_solucion, png=True, con_datos=False):
    """perfil_concentracion-promedio_{V}V.png (figura completa) y, con
    con_datos=True, el .txt de siempre (z, <c+>, <c->, <total>)."""
    sal, v = ctx["salt"], ctx["v_label"]
    rutas = []
    if png:
        rutas.append(pc.guardar_figura(
            crear_figura(datos_prom, ctx), ruta_solucion,
            f"section_avg_concentration_{v}V.png"))
    if con_datos:
        rutas.append(pc.exportar_txt(
            ruta_solucion, f"section_avg_concentration_{v}V.txt",
            [datos_prom["z_nm"], datos_prom["cp_prom"],
             datos_prom["cm_prom"], datos_prom["ct_prom"]],
            header=f"Average cross-section concentration | {v}V | {ctx['stem']}\n"
                   f"z_nm\t{sal['label_p']}_mM\t{sal['label_m']}_mM\ttotal_mM"))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos_prom, ctx = prep

    crear_figura(datos_prom, ctx)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()

    png = input("\nSave the image (PNG)? [Enter=No / y=Yes]: "
                ).strip().lower() in ("s", "si", "sí", "y")
    dat = input("Export the data (.txt)? [Enter=No / y=Yes]: "
                ).strip().lower() in ("s", "si", "sí", "y")
    if png or dat:
        guardar(datos_prom, ctx, ruta_solucion, png=png, con_datos=dat)
