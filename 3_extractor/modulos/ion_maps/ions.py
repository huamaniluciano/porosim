# -*- coding: utf-8 -*-
"""
Módulo: Mapa de Concentration 2D
Categoría: ion_maps

Genera mapas de color 2D de concentración de cation y anion (tricontourf).
Lee c0 y nombres de iones desde el _sim.json generado por el solver.

Funciona con o sin films. Cada film se dibuja como un borde:
    rojo  si la carga fija es positiva
    azul  si la carga fija es negativa

API pura (compartida entre procesar() de consola y la GUI):
    niveles_concentracion(c0, especie)                → niveles de color
    crear_figura(datos, ctx, especie, ...)            → 1 panel (cation/anion/total)
    crear_figura_comparativo(datos, ctx, ...)         → cation vs anion, 2 paneles

especie ∈ {"cation", "anion", "total"}.
datos = {"z_nm", "r_nm", "tri", "cp_mM", "cm_mM"} (arrays SIN espejar);
ctx   = porosim_comun.contexto_de(meta, stem, v_label).

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
# NIVELES DE COLOR (proporcionales a c0)
# =============================================================================
def niveles_concentracion(c0, especie="cation"):
    """Escala PROPORCIONAL a c0. Por-ion: c0/3 a 3·c0. Total (cation+anion,
    ≈2·c0 en bulk): 1.2·c0 a 5·c0. Así todos los plots comparten el intervalo
    relativo sin importar la c0 absoluta.

    ADVERTENCIA (gradiente): la escala se ancla a c0, único valor bulk. Cuando
    se implemente c_inlet ≠ c_outlet, "c0" deja de ser único y hay que
    redefinir el rango (p.ej. anclar a max(c_inlet, c_outlet) o al promedio)."""
    if especie == "total":
        return np.linspace(1.20 * c0, 5.00 * c0, 100)
    return np.linspace(c0 / 3.0, 3.0 * c0, 100)


def _campo_de(datos, especie):
    if especie == "cation":
        return datos["cp_mM"]
    if especie == "anion":
        return datos["cm_mM"]
    return datos["cp_mM"] + datos["cm_mM"]


def _label_de(ctx, especie):
    sal = ctx["salt"]
    if especie == "cation":
        return f"Catión {sal['label_p']}"
    if especie == "anion":
        return f"Anión {sal['label_m']}"
    return f"Iones totales ({sal['label_p']} + {sal['label_m']})"


# =============================================================================
# FIGURAS (las mismas para consola y GUI)
# =============================================================================
def crear_figura(datos, ctx, especie="cation", leyenda=True,
                 ylim=None, xlim=None, titulo=True, figsize=(14, 5)):
    """Mapa 2D de una especie (cation / anion / total) en un solo panel."""
    campo   = _campo_de(datos, especie)
    lbl_ion = _label_de(ctx, especie)
    niveles = niveles_concentracion(ctx["c0"], especie)

    z2, r2, tri2, c2 = pc.espejo(datos["z_nm"], datos["r_nm"], datos["tri"], campo)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.tricontourf(z2, r2, tri2, c2, levels=niveles, cmap="jet", extend="both")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Concentration [mM]")
    pc.marcar_colorbar(cbar)

    ax.set_aspect("equal")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_facecolor("0.93")
    if titulo:
        ax.set_title(f"2D map: {lbl_ion} — {ctx['v_label']}V  |  {ctx['stem']}")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")

    pc.guias_canal(ax, ctx)
    pc.rectangulos_films(ax, ctx, r_max_nm=r2.max(), con_labels=True)
    if leyenda and ctx["films"]:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return fig


def crear_figura_comparativo(datos, ctx, leyenda=True,
                             ylim=None, xlim=None, titulo=True,
                             figsize=(14, 9)):
    """Catión (arriba) vs anion (abajo), colorbar compartida. Es la figura del
    modo consola de ions.py y del 'comparativo lado a lado' de la GUI."""
    sal     = ctx["salt"]
    niveles = niveles_concentracion(ctx["c0"], "cation")
    z2p, r2p, tri2p, c2p = pc.espejo(datos["z_nm"], datos["r_nm"], datos["tri"],
                                     datos["cp_mM"])
    _,   _,   _,    c2m = pc.espejo(datos["z_nm"], datos["r_nm"], datos["tri"],
                                    datos["cm_mM"])

    fig, (ax_p, ax_m) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     constrained_layout=True)
    if titulo:
        fig.suptitle(f"Compared ionic concentration — {ctx['v_label']}V  |  "
                     f"{ctx['stem']}", fontsize=12)
    for ax_i, c_arr, lbl in [(ax_p, c2p, f"Catión {sal['label_p']}"),
                             (ax_m, c2m, f"Anión {sal['label_m']}")]:
        im = ax_i.tricontourf(z2p, r2p, tri2p, c_arr, levels=niveles,
                              cmap="jet", extend="both")
        ax_i.set_aspect("equal")
        if ylim is not None:
            ax_i.set_ylim(*ylim)
        if xlim is not None:
            ax_i.set_xlim(*xlim)
        ax_i.set_facecolor("0.93")
        ax_i.set_title(lbl)
        ax_i.set_ylabel("r [nm]")
        pc.guias_canal(ax_i, ctx)
        pc.rectangulos_films(ax_i, ctx, r_max_nm=r2p.max(),
                             con_labels=(ax_i is ax_p))
        if leyenda and ctx["films"] and ax_i is ax_p:
            ax_i.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax_m.set_xlabel("z [nm]")
    # Colorbar HORIZONTAL abajo: con set_aspect("equal") una colorbar vertical
    # obliga a matplotlib a anclar las tiras contra su lado (derecha) y deja un
    # hueco a la izquierda (el plot se ve "corrido"). Abajo roba alto, no ancho,
    # así los paneles ocupan todo el ancho y quedan centrados a cualquier zoom.
    cbar = fig.colorbar(im, ax=[ax_p, ax_m], orientation="horizontal",
                        fraction=0.05, pad=0.08)
    cbar.set_label("Concentration [mM]")
    pc.marcar_colorbar(cbar)
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None,
             titulo="MAPA DE CONCENTRACIÓN 2D"):
    """→ (datos, ctx) o None. datos = {z_nm, r_nm, tri, cp_mM, cm_mM}.
    `titulo` permite a total_ions reusar este preparar con su banner."""
    prep = pc.preparar_comun(ruta_solucion, titulo, v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    # Sin c0 real en el _sim.json las escalas (proporcionales a c0) mienten.
    if sol["meta"].get("simulation", {}).get("c0_mM") is None:
        print("❌ Missing 'c0_mM' in the _sim.json: the concentration maps "
              "are not reliable.")
        return None
    sal = ctx["salt"]
    dm  = sol["dm"]
    datos = {"z_nm": dm["z_nm"], "r_nm": dm["r_nm"], "tri": dm["tri"],
             "cp_mM": ctx["c0"] * np.exp(campos["up"]),
             "cm_mM": ctx["c0"] * np.exp(campos["um"])}
    print(f"    {sal['label_p']}: [{datos['cp_mM'].min():.2f}, "
          f"{datos['cp_mM'].max():.2f}] mM")
    print(f"    {sal['label_m']}: [{datos['cm_mM'].min():.2f}, "
          f"{datos['cm_mM'].max():.2f}] mM")
    return datos, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False):
    """Mapas limpios publication-ready, UN PNG POR ION:
    {stem}_mapa_{V}V_{ion}.png. Sin datos tabulares (con_datos se ignora)."""
    rutas = []
    if png:
        import re
        stem = pathlib.Path(ruta_solucion).stem
        for especie in ("cation", "anion"):
            figc = crear_figura(datos, ctx, especie=especie, leyenda=False,
                                titulo=False, figsize=(10, 4))
            lbl  = (ctx["salt"]["label_p"] if especie == "cation"
                    else ctx["salt"]["label_m"])
            slug = re.sub(r"[^0-9A-Za-z]+", "", lbl) or especie
            rutas.append(pc.guardar_figura(
                figc, ruta_solucion,
                f"{stem}_mapa_{ctx['v_label']}V_{slug}.png", limpio=True))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep

    print(">>> Generating 2D maps...")
    crear_figura_comparativo(datos, ctx)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()

    resp = input("\nSave clean maps (one PNG per ion, content + "
                 "scale only)? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion)
    else:
        print("    (no images saved)")
