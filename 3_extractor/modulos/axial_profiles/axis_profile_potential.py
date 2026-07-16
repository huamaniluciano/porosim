# -*- coding: utf-8 -*-
"""
Módulo: Perfil Axial de Potential
Categoría: axial_profiles

Muestra el perfil del potencial eléctrico phi (en Volts) a lo largo del eje
de simetría (r = 0) para un voltaje elegido por el usuario.
Lee la temperatura T_K, el canal y los films desde el _sim.json.

El perfil se toma de los NODOS REALES de la malla sobre el eje (no por
muestreo en z arbitrarios): así ningún rasgo fino (p.ej. el salto Donnan) se
pierde por caer entre dos puntos, y la resolución sigue el refinamiento de la
malla. La solución es P1 → unir los nodos con rectas reproduce EXACTAMENTE la
solución de elementos finitos en el eje.

Funciona con o sin films. Cada film se marca como una banda roja (carga +) o
azul (−) y, en equilibrio (V≈0), su salto Donnan analítico como línea horizontal.

API pura (compartida entre procesar() de consola y la GUI):
    dibujar_potencial_axial(ax, datos, ctx, leyenda=True)   → pinta sobre un Axes
    crear_figura(datos, ctx, ...)                           → Figure de 1 panel

datos = {"z_axis_nm", "phi_ax_V"}; ctx = porosim_comun.contexto_de(...).

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


# =============================================================================
# TRAZADO (primitiva compartida: pinta sobre un Axes ya creado)
# =============================================================================
def dibujar_potencial_axial(ax, datos, ctx, leyenda=True, con_xlabel=True):
    """Pinta φ(r=0) vs z sobre `ax`. La usan crear_figura() y la vista
    combinada de la GUI (potencial + concentración apilados)."""
    z_axis = datos["z_axis_nm"]
    phi_ax = datos["phi_ax_V"]
    v_num  = ctx.get("v_num", 0.0) or 0.0

    ax.plot(z_axis, phi_ax, "-", color="darkorange", lw=2.2,
            label=r"Potential $\phi(r=0)$")
    ax.axhline(0.0, color="black", ls=":", lw=1, alpha=0.5)
    ax.axhline(v_num, color="blue", ls="--", lw=1, alpha=0.4,
               label=f"Applied voltage ({v_num:+.2f} V)")

    pc.guias_canal(ax, ctx, con_label=leyenda)
    pc.bandas_films(ax, ctx, con_labels=leyenda)

    # Salto Donnan analítico: solo comparable en equilibrio → se dibuja a V≈0.
    if abs(v_num) < 1e-9:
        for film in ctx["films"]:
            if film.get("phi_D_mV") is not None:
                color = "red" if film["rho"] > 0 else "blue"
                ax.axhline(film["phi_D_mV"] / 1e3, color=color, ls="-.", lw=1.5,
                           alpha=0.6,
                           label=f"Donnan {film['side']} ({film['phi_D_mV']:.1f} mV)")

    ax.set_ylabel("Potential [V]")
    if con_xlabel:
        ax.set_xlabel("z [nm]")
    ax.grid(True, alpha=0.3)
    if leyenda:
        ax.legend(fontsize=8, loc="best")


def crear_figura(datos, ctx, leyenda=True, titulo=True, figsize=(11, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    if titulo:
        fig.suptitle(f"Potential Profile on the Axis (r=0) — {ctx['v_label']}V  |  "
                     f"{ctx['stem']}", fontsize=12)
    dibujar_potencial_axial(ax, datos, ctx, leyenda=leyenda)
    ax.set_title("Electric Potential along the Nanopore")
    fig.tight_layout()
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """→ (datos, ctx) o None. datos = {z_axis_nm, phi_ax_V} (nodos del eje)."""
    prep = pc.preparar_comun(ruta_solucion, "AXIAL ELECTRIC POTENTIAL PROFILE",
                             v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    print(f"  Temperature: {ctx['T_K']} K  |  V_thermal = {ctx['V_T']*1e3:.2f} mV")

    dm    = sol["dm"]
    ai    = dm["axis_idx"]
    phi_V = campos["phi_adim"] * ctx["V_T"]
    print(f">>> Profile taken from {len(ai)} axis nodes (no arbitrary sampling)")
    return {"z_axis_nm": dm["z_nm"][ai], "phi_ax_V": phi_V[ai]}, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False):
    """perfil_potencial_{V}V.png (figura completa, con ejes) y, con
    con_datos=True, el .txt de siempre (z_nm, phi_V)."""
    v, rutas = ctx["v_label"], []
    if png:
        rutas.append(pc.guardar_figura(crear_figura(datos, ctx), ruta_solucion,
                                       f"potential_profile_{v}V.png"))
    if con_datos:
        rutas.append(pc.exportar_txt(
            ruta_solucion, f"potential_profile_{v}V.txt",
            [datos["z_axis_nm"], datos["phi_ax_V"]],
            header=f"Axial potential profile (r=0) | {v}V | {ctx['stem']}\n"
                   f"z_nm\tphi_V"))
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
    dat = input("Export the data (.txt)? [Enter=No / y=Yes]: "
                ).strip().lower() in ("s", "si", "sí", "y")
    if png or dat:
        guardar(datos, ctx, ruta_solucion, png=png, con_datos=dat)
