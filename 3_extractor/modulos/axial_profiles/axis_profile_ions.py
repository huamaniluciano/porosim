# -*- coding: utf-8 -*-
"""
Módulo: Perfil de Concentration en el Eje
Categoría: axial_profiles

Muestra los perfiles axiales (r=0) de cation, anion e iones totales para un
voltaje elegido por el usuario. Lee c0, nombres de iones, canal y films desde
el _sim.json.

El perfil se toma de los NODOS REALES de la malla sobre el eje (no por muestreo
en z arbitrarios): así ningún rasgo fino se pierde por caer entre dos puntos, y
la resolución sigue el refinamiento de la malla.

Funciona con o sin films. Cada film se marca como una banda roja (carga +) o
azul (−).

API pura (compartida entre procesar() de consola y la GUI):
    dibujar_iones_axial(ax, datos, ctx, curvas, leyenda=True)  → pinta sobre un Axes
    crear_figura(datos, ctx, curvas, ...)                      → Figure de 1 panel

curvas: subconjunto de ("cation", "anion", "total").
datos = {"z_axis_nm", "cp_ax_mM", "cm_ax_mM", "ct_ax_mM"};
ctx   = porosim_comun.contexto_de(...).

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc

CURVAS_DEFAULT = ("total", "anion", "cation")


# =============================================================================
# TRAZADO (primitiva compartida: pinta sobre un Axes ya creado)
# =============================================================================
def dibujar_iones_axial(ax, datos, ctx, curvas=CURVAS_DEFAULT,
                        leyenda=True, con_xlabel=True):
    """Pinta las concentraciones axiales seleccionadas sobre `ax`. La usan
    crear_figura() y la vista combinada de la GUI."""
    z_axis = datos["z_axis_nm"]
    cp, cm, ct = datos["cp_ax_mM"], datos["cm_ax_mM"], datos["ct_ax_mM"]
    c0, sal = ctx["c0"], ctx["salt"]

    if "total" in curvas:
        ax.plot(z_axis, ct, "-", color="darkorchid", lw=2, label="Total ions")
    if "anion" in curvas:
        ax.plot(z_axis, cm, "--", color="steelblue", lw=2.2, label=sal["label_m"])
    if "cation" in curvas:
        ax.plot(z_axis, cp, "-", color="crimson", lw=1.6, label=sal["label_p"])

    ax.axhline(2 * c0, color="black", ls=":", lw=1, label=f"Bulk total ({2*c0:.0f} mM)")
    ax.axhline(c0, color="gray", ls=":", lw=1, alpha=0.6, label=f"Bulk ion ({c0:.0f} mM)")
    pc.guias_canal(ax, ctx)
    pc.bandas_films(ax, ctx, con_labels=False)

    max_y = max(ct.max() if len(ct) else c0, c0 * 2) * 1.15
    ax.set_ylim(0, max_y)
    if con_xlabel:
        ax.set_xlabel("z [nm]")
    ax.set_ylabel("Concentration [mM]")
    ax.grid(True, alpha=0.3)
    if leyenda:
        ax.legend(fontsize=8, loc="best")


def crear_figura(datos, ctx, curvas=CURVAS_DEFAULT, leyenda=True,
                 titulo=True, figsize=(11, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    if titulo:
        fig.suptitle(f"Ion profile on the Central Axis (r=0) — {ctx['v_label']}V  |  "
                     f"{ctx['stem']}", fontsize=12)
    dibujar_iones_axial(ax, datos, ctx, curvas=curvas, leyenda=leyenda)
    fig.tight_layout()
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """→ (datos, ctx) o None.
    datos = {z_axis_nm, cp_ax_mM, cm_ax_mM, ct_ax_mM} (nodos del eje)."""
    prep = pc.preparar_comun(ruta_solucion, "AXIAL CONCENTRATION PROFILE",
                             v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    dm = sol["dm"]
    ai = dm["axis_idx"]
    cp = ctx["c0"] * np.exp(campos["up"][ai])
    cm = ctx["c0"] * np.exp(campos["um"][ai])
    print(f">>> Profile taken from {len(ai)} axis nodes (no arbitrary sampling)")
    return {"z_axis_nm": dm["z_nm"][ai], "cp_ax_mM": cp, "cm_ax_mM": cm,
            "ct_ax_mM": cp + cm}, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False):
    """perfil_iones_{V}V.png (figura completa, con ejes) y, con
    con_datos=True, el .txt + .xlsx de siempre (z, c+, c-, total)."""
    sal, v = ctx["salt"], ctx["v_label"]
    z, cp, cm, ct = (datos["z_axis_nm"], datos["cp_ax_mM"],
                     datos["cm_ax_mM"], datos["ct_ax_mM"])
    rutas = []
    if png:
        rutas.append(pc.guardar_figura(crear_figura(datos, ctx), ruta_solucion,
                                       f"ion_profile_{v}V.png"))
    if con_datos:
        rutas.append(pc.exportar_txt(
            ruta_solucion, f"ion_profile_{v}V.txt", [z, cp, cm, ct],
            header=f"Axial concentration profile (r=0) | {v}V | {ctx['stem']}\n"
                   f"z_nm\t{sal['label_p']}_mM\t{sal['label_m']}_mM\ttotal_mM"))
        x = pc.exportar_xlsx(
            ruta_solucion, f"ion_profile_{v}V.xlsx",
            {"z_nm": z, f"{sal['label_p']}_mM": cp,
             f"{sal['label_m']}_mM": cm, "total_mM": ct})
        if x is not None:
            rutas.append(x)
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep

    print(">>> Generating plot...")
    crear_figura(datos, ctx)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()

    png = input("\nSave the image (PNG)? [Enter=No / y=Yes]: "
                ).strip().lower() in ("s", "si", "sí", "y")
    dat = input("Export the data (.txt + .xlsx)? [Enter=No / y=Yes]: "
                ).strip().lower() in ("s", "si", "sí", "y")
    if png or dat:
        guardar(datos, ctx, ruta_solucion, png=png, con_datos=dat)
