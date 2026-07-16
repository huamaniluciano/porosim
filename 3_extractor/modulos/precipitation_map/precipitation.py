# -*- coding: utf-8 -*-
"""
Módulo: Precipitation Map 2D con interactividad
Categoría: precipitation_map

Grafica la zona de precipitación de una sal parcialmente soluble en base a la
ecuación de Davies. TextBox interactivo permite modificar el factor de
sobresaturación en tiempo real.

La sal, su Kps y su solubilidad se leen del bloque "salt" del _sim.json (fuente
única de verdad del catálogo de sales del solver). El módulo NO hardcodea el
Kps ni el nombre de la sal: si mañana se agrega otra sal parcialmente soluble
al catálogo, este módulo la procesa sin cambios.

aplica(meta): este módulo SOLO tiene sentido para sales parcialmente solubles
(soluble=False y Kps definido). El ejecutor lo usa para no ofrecerlo en el menú
cuando la solución abierta es de una sal totalmente soluble (ej. KCl, NaCl).

Funciona con o sin films. Cada film se marca como una banda:
    roja  si la carga fija es positiva
    azul  si la carga fija es negativa

API pura (la usan procesar() de consola Y la GUI — editá acá y repercute en
las dos):
    calc_gamma_davies(I_molar, z_p, z_m)          → γ± (Davies, cap 0.5 M)
    calcular_precipitacion(cp, cm, z_p, z_m, Kps, factor)
                                                  → {mapa_bin, Q_act, n_precip, n_total}
    crear_figura(datos, ctx, factor, ...)         → matplotlib Figure

datos = {"z_nm", "r_nm", "tri", "cp_mM", "cm_mM"} (arrays SIN espejar);
ctx   = porosim_comun.contexto_de(meta, stem, v_label).

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc

# Factor de sobresaturación del PAPER (precipita donde Q_act > 2·Kps, S≈1.4).
# DECISIÓN FIJADA (ver ../../PENDIENTES.md): es el default del batch, de la
# GUI (slider) y del TextBox de consola. Cambiarlo SOLO acá.
FACTOR_PAPER = 2.0


# =============================================================================
# FÍSICA (pura NumPy)
# =============================================================================
def calc_gamma_davies(I_molar, z_p, z_m):
    """
    Coeficiente de actividad medio según Davies (válido hasta ~I = 0.5 M).
    Por encima de 0.5 M se "capea" la I para no salirse del rango de validez.

    DECISIÓN FIJADA (ver ../../PENDIENTES.md): el modelo de actividades del
    paper es ESTE (Davies + cap 0.5 M). No cambiarlo sin recalcular todo el
    barrido de precipitación publicado.
    """
    I_safe = np.maximum(I_molar, 1e-12)
    I_eval = np.minimum(I_safe, 0.5)  # Cap de Davies a 0.5 M
    sqrt_I = np.sqrt(I_eval)

    # log10(gamma) = -A · |z_p · z_m| · [ sqrt(I)/(1 + sqrt(I)) - 0.3·I ]
    # A = 0.51 a 25 °C
    log_gamma = -0.51 * abs(z_p * z_m) * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I_eval)
    return 10.0 ** log_gamma


def calcular_precipitacion(cp_mM, cm_mM, z_p, z_m, Kps_M2, factor):
    """Mapa binario de precipitación: 1 donde Q_act > factor·Kps.
    Q_act = γ±² · c⁺ · c⁻ (en M²), con γ± de Davies sobre la fuerza iónica."""
    I_M   = 0.5 * (cp_mM * z_p**2 + cm_mM * z_m**2) / 1000.0
    gamma = calc_gamma_davies(I_M, z_p, z_m)
    Q_act = (gamma**2) * (cp_mM / 1000.0) * (cm_mM / 1000.0)
    mapa  = np.where(Q_act > factor * Kps_M2, 1.0, 0.0)
    return {"mapa_bin": mapa, "Q_act": Q_act,
            "n_precip": int(mapa.sum()), "n_total": mapa.size}


def aplica(meta):
    """
    ¿Tiene sentido este módulo para la solución abierta? Solo para sales
    parcialmente solubles (soluble=False y Kps definido). El ejecutor lo
    consulta para no listar el módulo cuando la sal es totalmente soluble.

    _sim.json viejos (sin bloque "salt") → True, para no ocultar el módulo en
    soluciones previas a la introducción del catálogo de sales.
    """
    sal = meta.get("simulation", {}).get("salt")
    if sal is None:
        return True
    return sal.get("soluble") is False and sal.get("Ksp_M2") is not None


# =============================================================================
# FIGURA (la misma para consola y GUI)
# =============================================================================
CMAP_BIN = ListedColormap(["#80CCFF", "#FF0000"])


def leyenda_precipitacion():
    return [Patch(facecolor="#80CCFF", edgecolor="k",
                  label="Soluble (Q_act < threshold)"),
            Patch(facecolor="#FF0000", edgecolor="k",
                  label="Precipitates (Q_act > threshold)")]


def dibujar_mapa(ax, z2, r2, tri2, bin2, ctx, factor,
                 leyenda=True, xlim="auto", ylim=None, info=None):
    """Dibuja el mapa binario (ya espejado) sobre un Axes existente.
    xlim="auto" → zoom canal ±150 nm; None → dominio completo; (a,b) → manual."""
    ax.tricontourf(z2, r2, tri2, bin2, levels=[-0.5, 0.5, 1.5], cmap=CMAP_BIN)
    ax.set_aspect("equal")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_facecolor("0.85")   # fuera del dominio (membrana) = gris

    z_tip, z_base = ctx.get("z_tip"), ctx.get("z_base")
    if xlim == "auto":
        if z_tip is not None and z_base is not None:
            ax.set_xlim(min(z_tip, z_base) * 1e9 - 150,
                        max(z_tip, z_base) * 1e9 + 150)
    elif xlim is not None:
        ax.set_xlim(*xlim)

    if info is not None:
        pct = 100.0 * info["n_precip"] / info["n_total"]
        ax.set_title(f"Precipitates: {info['n_precip']}/{info['n_total']} ({pct:.2f}%)")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")
    ax.grid(True, alpha=0.15)

    pc.guias_canal(ax, ctx)
    pc.bandas_films(ax, ctx, con_labels=False)

    if leyenda:
        ax.legend(handles=leyenda_precipitacion(), loc="upper right",
                  fontsize=8, framealpha=0.95, title=f"factor = {factor:g} × Kps")


def crear_figura(datos, ctx, factor=FACTOR_PAPER, leyenda=True,
                 xlim="auto", ylim=(-500, 500), titulo=True, figsize=(11, 7)):
    """Figura canónica del módulo a partir de arrays en memoria.
    Deja los conteos en fig.info_precipitacion = {n_precip, n_total}."""
    sal = ctx["salt"]
    Kps = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2

    res = calcular_precipitacion(datos["cp_mM"], datos["cm_mM"],
                                 sal["z_p"], sal["z_m"], Kps, factor)
    z2, r2, tri2, bin2 = pc.espejo(datos["z_nm"], datos["r_nm"],
                                   datos["tri"], res["mapa_bin"])

    fig, ax = plt.subplots(figsize=figsize)
    if titulo:
        fig.suptitle(f"Precipitation Map ({sal['name']}) — {ctx['v_label']}V\n"
                     f"Kps = {Kps:.2e} M²  |  {ctx['stem']}", fontsize=11)
    dibujar_mapa(ax, z2, r2, tri2, bin2, ctx, factor,
                 leyenda=leyenda, xlim=xlim, ylim=ylim, info=res)
    fig.tight_layout()
    fig.info_precipitacion = {"n_precip": res["n_precip"],
                              "n_total": res["n_total"]}
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """→ (datos, ctx) o None. datos = {z_nm, r_nm, tri, cp_mM, cm_mM}.
    Aborta ANTES de cargar la malla si la sal es totalmente soluble
    (aplica() = False: el mapa no tiene sentido físico)."""
    meta_previa = (sol["meta"] if sol is not None
                   else pc.cargar_meta(pathlib.Path(ruta_solucion).parent))
    if not aplica(meta_previa):
        sal_nombre = meta_previa.get("simulation", {}).get("salt", {}).get("name", "?")
        print(f"  [NOTE] '{sal_nombre}' is a fully soluble salt: it does not "
              f"precipitate. This module does not apply.")
        return None

    prep = pc.preparar_comun(ruta_solucion, "2D PRECIPITATION MAP",
                             v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    if sol["meta"].get("simulation", {}).get("c0_mM") is None:
        print("❌ Missing 'c0_mM' in the _sim.json simulation.")
        return None

    sal = ctx["salt"]
    Kps = sal["Ksp_M2"]
    if Kps is None:
        print(f"  [NOTE] '{sal['name']}' does not define Ksp; using "
              f"{pc.KPS_LEGACY_M2:.2e} M² (KClO4).")
        Kps = pc.KPS_LEGACY_M2
    print(f"  Ksp = {Kps:.2e} M²")

    dm = sol["dm"]
    datos = {"z_nm": dm["z_nm"], "r_nm": dm["r_nm"], "tri": dm["tri"],
             "cp_mM": ctx["c0"] * np.exp(campos["up"]),
             "cm_mM": ctx["c0"] * np.exp(campos["um"])}
    return datos, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False,
            factor=FACTOR_PAPER):
    """Mapa limpio publication-ready: {stem}_precipitacion_{V}V_f{factor}.png
    (la leyenda soluble/precipita se preserva: es la "escala" del binario).
    con_datos=True → .txt con el conteo (factor, Kps, nodos que precipitan)."""
    sal = ctx["salt"]
    Kps = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2
    res = calcular_precipitacion(datos["cp_mM"], datos["cm_mM"],
                                 sal["z_p"], sal["z_m"], Kps, factor)
    stem, v = pathlib.Path(ruta_solucion).stem, ctx["v_label"]
    rutas = []
    if png:
        figc = crear_figura(datos, ctx, factor=factor, titulo=False,
                            figsize=(10, 4))
        rutas.append(pc.guardar_figura(
            figc, ruta_solucion,
            f"{stem}_precipitation_{v}V_f{factor:g}.png", limpio=True))
    if con_datos:
        pct = 100.0 * res["n_precip"] / res["n_total"]
        fpath = (pathlib.Path(ruta_solucion).parent /
                 f"{stem}_precipitation_{v}V_f{factor:g}.txt")
        fpath.write_text(
            f"# Precipitation ({sal['name']}) | {v}V | {stem}\n"
            f"# Q_act = gamma±²·c+·c- (Davies) > factor·Kps\n"
            f"factor_x_Kps\t{factor:g}\n"
            f"Kps_M2\t{Kps:.6e}\n"
            f"nodos_precipitan\t{res['n_precip']}\n"
            f"nodos_totales\t{res['n_total']}\n"
            f"porcentaje\t{pct:.4f}\n", encoding="utf-8")
        print(f"    ✓ Saved: {fpath}")
        rutas.append(fpath)
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo, TextBox de factor en vivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep
    sal       = ctx["salt"]
    Kps_molar = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2

    # ── Gráfico interactivo con TextBox ──────────────────────────────────────
    print(">>> Generating interactive graphical interface...")
    from matplotlib.widgets import TextBox

    factor_inicial = FACTOR_PAPER
    estado = {"factor": factor_inicial}

    fig = crear_figura(datos, ctx, factor=factor_inicial)
    ax  = fig.axes[0]
    fig.subplots_adjust(bottom=0.22)   # espacio inferior para el TextBox

    # Espejo precalculado UNA vez; cada redibujo solo recalcula el mapa binario
    z2, r2, tri2, _ = pc.espejo(datos["z_nm"], datos["r_nm"], datos["tri"],
                                datos["cp_mM"])

    def refrescar_grafico(factor):
        res = calcular_precipitacion(datos["cp_mM"], datos["cm_mM"],
                                     sal["z_p"], sal["z_m"], Kps_molar, factor)
        bin2 = np.concatenate([res["mapa_bin"], res["mapa_bin"]])
        ax.clear()
        dibujar_mapa(ax, z2, r2, tri2, bin2, ctx, factor, info=res)
        fig.canvas.draw_idle()

    ax_box   = fig.add_axes([0.45, 0.06, 0.15, 0.045])
    text_box = TextBox(ax_box, "Factor (× Ksp): ", initial=str(factor_inicial))

    def on_submit(text_val):
        try:
            val = float(text_val)
            if val <= 0:
                print("⚠️ The supersaturation factor must be greater than 0.")
                return
            refrescar_grafico(val)
            estado["factor"] = val
            print(f">>> Map updated for factor_sob = {val}")
        except ValueError:
            print("⚠️ Invalid input. Enter a numeric value.")

    text_box.on_submit(on_submit)
    fig.text_box_reference = text_box   # que no lo recoja el GC

    print(">>> Opening interactive window. Close it to return to the menu.")
    plt.show()

    # ── Guardado limpio opcional (misma figura canónica, sin ejes) ──────────
    # Se guarda con el ÚLTIMO factor aplicado en el TextBox (estado["factor"]).
    resp = input("\nSave clean map (content + scale only, no axes)? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion, factor=estado["factor"])
    else:
        print("    (no image saved)")
