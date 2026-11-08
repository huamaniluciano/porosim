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
    saturacion(cp, cm, z_p, z_m, Kps)             → Ω = Q_act/Ksp en cada nodo
    clasificar_saturacion(cp, cm, z_p, z_m, Kps, factor)
                                                  → 0 subsaturada | 1 sobresaturada
                                                    sin nuclear | 2 precipita
    s_max(datos, Kps, z_p, z_m)                   → max Ω del dominio (un voltaje)
    voltaje_umbral(V, S_max, factor)              → |V| de aparición por polaridad
    crear_figura(datos, ctx, factor, estilo, ...) → matplotlib Figure
    nombre_base(stem, v_label, factor, estilo)    → nombre de archivo (sin extensión)

El barrido completo (max Ω vs V y voltaje de aparición) es el módulo global
precipitation_onset.py, que usa s_max() y voltaje_umbral() de acá.

`factor` es el umbral S de sobresaturación: precipita donde Q_act > S·Ksp.
Default FACTOR_PAPER; en batch se cambia con --opt factor=10 (v1.1.0).
`estilo` del mapa (v1.1.0): "binario" (el de siempre), "ternario" (separa la
zona sobresaturada que todavía no nuclea, Ksp < Q_act ≤ S·Ksp) o "log10"
(log₁₀ Ω continuo, con la línea del umbral).

datos = {"z_nm", "r_nm", "tri", "cp_mM", "cm_mM"} (arrays SIN espejar);
ctx   = porosim_comun.contexto_de(meta, stem, v_label).

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc

# Umbral S de sobresaturación por defecto: precipita donde Q_act > 2·Kps
# (S ≈ 1.4 en concentración), el de las figuras de v1.0.0.
# DECISIÓN FIJADA (ver ../../PENDIENTES.md): es el default del batch, de la
# GUI y del TextBox de consola. Desde v1.1.0 S es un parámetro (batch:
# --opt factor=10; GUI y consola: campo numérico); el DEFAULT se cambia SOLO acá.
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
    soluciones previas a la introducción del catálogo de sales. Una mezcla
    (v1.1.0) → False: Q = a+·a− supone el par catión/anión de UNA sal.
    """
    if not pc.aplica_binaria(meta):
        return False
    sal = meta.get("simulation", {}).get("salt")
    if sal is None:
        return True
    return sal.get("soluble") is False and sal.get("Ksp_M2") is not None


def saturacion(cp_mM, cm_mM, z_p, z_m, Kps_M2):
    """Ω = Q_act/Ksp en cada nodo (Ω = 1: saturada; Davies, como el mapa)."""
    return calcular_precipitacion(cp_mM, cm_mM, z_p, z_m, Kps_M2, 1.0)["Q_act"] / Kps_M2


def clasificar_saturacion(cp_mM, cm_mM, z_p, z_m, Kps_M2, factor):
    """0: Q_act ≤ Ksp (subsaturada, un sólido se disolvería) | 1: Ksp < Q_act ≤
    factor·Ksp (sobresaturada, pero bajo el umbral de nucleación) | 2: Q_act >
    factor·Ksp (precipita: es exactamente el 1 del mapa binario)."""
    Q = calcular_precipitacion(cp_mM, cm_mM, z_p, z_m, Kps_M2, factor)["Q_act"]
    return np.where(Q > factor * Kps_M2, 2.0, np.where(Q > Kps_M2, 1.0, 0.0))


def s_max(datos, Kps, z_p, z_m):
    """max Ω = max(Q_act/Ksp) sobre todo el dominio, para UN voltaje."""
    return float(np.max(saturacion(datos["cp_mM"], datos["cm_mM"], z_p, z_m, Kps)))


def voltaje_umbral(V, smax, factor):
    """Primer |V| (interpolado linealmente entre voltajes calculados) en que
    max Ω supera `factor`, por polaridad → {"+": |V| o None, "-": |V| o None}.
    Si ya lo supera en el |V| más chico de la rama, devuelve ese |V|."""
    V, smax = np.asarray(V, dtype=float), np.asarray(smax, dtype=float)
    out = {}
    for signo, m in (("+", V >= 0), ("-", V <= 0)):
        v, s = np.abs(V[m]), smax[m]
        orden = np.argsort(v)
        v, s = v[orden], s[orden]
        idx = np.nonzero(s > factor)[0]
        if len(idx) == 0:
            out[signo] = None
        elif idx[0] == 0:
            out[signo] = float(v[0])
        else:
            i = idx[0]
            out[signo] = float(v[i - 1] + (factor - s[i - 1]) * (v[i] - v[i - 1])
                               / (s[i] - s[i - 1]))
    return out


# =============================================================================
# FIGURA (la misma para consola y GUI)
# =============================================================================
CMAP_BIN = ListedColormap(["#80CCFF", "#FF0000"])
# Ternario: los colores del binario + ámbar para la zona sobresaturada que
# todavía no nuclea (el binario la pinta de azul, como "soluble").
COLOR_META    = "#FFC857"
CMAP_TERNARIO = ListedColormap(["#80CCFF", COLOR_META, "#FF0000"])
ESTILOS = ("binario", "ternario", "log10")
_ALIAS_ESTILO = {"binary": "binario", "ternary": "ternario", "log": "log10",
                 "log10s": "log10", "saturation": "log10"}
LOG10_RANGO = 3.0          # escala fija ±3 décadas de Ω: comparable entre voltajes


def estilo_valido(estilo):
    """'binary'/'ternary'/'log' (inglés) o los nombres de ESTILOS → nombre interno."""
    e = str(estilo).strip().lower()
    e = _ALIAS_ESTILO.get(e, e)
    if e not in ESTILOS:
        raise ValueError(f"estilo '{estilo}' unknown: use one of {ESTILOS} "
                         f"(or binary/ternary/log)")
    return e


def nombre_base(stem, v_label, factor, estilo="binario"):
    """Nombre (sin extensión) del mapa de cada estilo, el mismo en batch y GUI.
    binario → {stem}_precipitation_{V}V_f{S} (el de v1.0.0)."""
    sufijo = {"binario": "precipitation", "ternario": "saturation_ternary",
              "log10": "saturation_log10"}[estilo_valido(estilo)]
    return f"{stem}_{sufijo}_{v_label}V_f{float(factor):g}"


def leyenda_precipitacion():
    return [Patch(facecolor="#80CCFF", edgecolor="k",
                  label="Soluble (Q_act < threshold)"),
            Patch(facecolor="#FF0000", edgecolor="k",
                  label="Precipitates (Q_act > threshold)")]


def leyenda_ternaria():
    return [Patch(facecolor="#80CCFF", edgecolor="k", label="Undersaturated (Q_act ≤ Ksp)"),
            Patch(facecolor=COLOR_META, edgecolor="k",
                  label="Supersaturated, below S (Ksp < Q_act ≤ S·Ksp)"),
            Patch(facecolor="#FF0000", edgecolor="k", label="Precipitates (Q_act > S·Ksp)")]


def leyenda_log10(factor):
    """Las dos isolíneas del estilo log10 (la escala de color va en la barra)."""
    return [Line2D([], [], color="k", lw=0.9, ls="--",
                   label=f"Q_act = S·Ksp (threshold, S = {factor:g})"),
            Line2D([], [], color="k", lw=0.9, ls=":", label="Q_act = Ksp (saturation)")]


def valores_mapa(datos, ctx, factor, estilo="binario"):
    """Campo nodal (sin espejar) que pinta cada estilo + los conteos:
    binario → 0/1, ternario → 0/1/2, log10 → log₁₀ Ω."""
    sal = ctx["salt"]
    Kps = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2
    res = calcular_precipitacion(datos["cp_mM"], datos["cm_mM"],
                                 sal["z_p"], sal["z_m"], Kps, factor)
    estilo = estilo_valido(estilo)
    if estilo == "binario":
        valores = res["mapa_bin"]
    elif estilo == "ternario":
        valores = clasificar_saturacion(datos["cp_mM"], datos["cm_mM"],
                                        sal["z_p"], sal["z_m"], Kps, factor)
    else:
        valores = np.log10(np.maximum(res["Q_act"] / Kps, 1e-300))
    res["omega_max"] = float(np.max(res["Q_act"]) / Kps)
    res["n_sobresat"] = int(np.count_nonzero(res["Q_act"] > Kps))
    return valores, res


def dibujar_mapa(ax, z2, r2, tri2, bin2, ctx, factor,
                 leyenda=True, xlim="auto", ylim=None, info=None, estilo="binario",
                 colorbar=True):
    """Dibuja el mapa (ya espejado) sobre un Axes existente. `bin2` son los
    valores de `valores_mapa` para ese estilo (0/1 en el binario).
    xlim="auto" → zoom canal ±150 nm; None → dominio completo; (a,b) → manual.
    colorbar=False (log10) → no agrega otra barra: la escala ±3 décadas es fija,
    así que al redibujar con otro S la barra que ya está sigue valiendo."""
    estilo = estilo_valido(estilo)
    mapeable = None
    if estilo == "binario":
        ax.tricontourf(z2, r2, tri2, bin2, levels=[-0.5, 0.5, 1.5], cmap=CMAP_BIN)
    elif estilo == "ternario":
        ax.tricontourf(z2, r2, tri2, bin2, levels=[-0.5, 0.5, 1.5, 2.5],
                       cmap=CMAP_TERNARIO)
    else:
        niveles = np.linspace(-LOG10_RANGO, LOG10_RANGO, 25)
        mapeable = ax.tricontourf(z2, r2, tri2, np.clip(bin2, -LOG10_RANGO, LOG10_RANGO),
                                  levels=niveles, cmap="RdBu_r", extend="neither")
        log_f = np.log10(factor) if factor > 0 else None
        for nivel, estilo_l in ((0.0, ":"), (log_f, "--")):
            if nivel is not None and bin2.min() < nivel < bin2.max():
                ax.tricontour(z2, r2, tri2, bin2, levels=[nivel], colors="k",
                              linewidths=0.9, linestyles=estilo_l)
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
        titulo = f"Precipitates: {info['n_precip']}/{info['n_total']} ({pct:.2f}%)"
        if estilo != "binario" and "omega_max" in info:
            titulo += f"  |  max Q_act/Ksp = {info['omega_max']:.3g}"
        ax.set_title(titulo)
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")
    ax.grid(True, alpha=0.15)

    pc.guias_canal(ax, ctx)
    pc.bandas_films(ax, ctx, con_labels=False)

    if mapeable is not None and colorbar:
        cbar = ax.figure.colorbar(mapeable, ax=ax, shrink=0.85)
        cbar.set_label("log₁₀(Q_act / Ksp)")
        pc.ticks_colorbar(cbar, -LOG10_RANGO, LOG10_RANGO)
        for nivel, estilo_l in ((0.0, ":"), (np.log10(factor) if factor > 0 else None, "--")):
            if nivel is not None and -LOG10_RANGO < nivel < LOG10_RANGO:
                cbar.ax.axhline(nivel, color="k", lw=1.0, ls=estilo_l)
    if leyenda:
        if estilo == "log10":
            ax.legend(handles=leyenda_log10(factor), loc="upper right", fontsize=8,
                      framealpha=0.95)
        else:
            ax.legend(handles=(leyenda_precipitacion() if estilo == "binario"
                               else leyenda_ternaria()),
                      loc="upper right", fontsize=8, framealpha=0.95,
                      title=f"S = {factor:g} × Ksp")


def crear_figura(datos, ctx, factor=FACTOR_PAPER, leyenda=True,
                 xlim="auto", ylim=(-500, 500), titulo=True, figsize=(11, 7),
                 estilo="binario"):
    """Figura canónica del módulo a partir de arrays en memoria.
    Deja los conteos en fig.info_precipitacion = {n_precip, n_total,
    n_sobresat, omega_max}."""
    sal = ctx["salt"]
    Kps = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2
    estilo = estilo_valido(estilo)

    valores, res = valores_mapa(datos, ctx, factor, estilo)
    z2, r2, tri2, val2 = pc.espejo(datos["z_nm"], datos["r_nm"],
                                   datos["tri"], valores)

    fig, ax = plt.subplots(figsize=figsize)
    if titulo:
        nombre = {"binario": "Precipitation Map", "ternario": "Saturation Map",
                  "log10": "Saturation Q_act/Ksp"}[estilo]
        fig.suptitle(f"{nombre} ({sal['name']}) — {ctx['v_label']}V\n"
                     f"Kps = {Kps:.2e} M²  |  {ctx['stem']}", fontsize=11)
    dibujar_mapa(ax, z2, r2, tri2, val2, ctx, factor,
                 leyenda=leyenda, xlim=xlim, ylim=ylim, info=res, estilo=estilo)
    fig.tight_layout()
    fig.info_precipitacion = {"n_precip": res["n_precip"],
                              "n_total": res["n_total"],
                              "n_sobresat": res["n_sobresat"],
                              "omega_max": res["omega_max"]}
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
    if not pc.aplica_binaria(meta_previa):
        pc.exigir_binaria(meta_previa)          # prints why (mixture / no c0)
        return None
    if not aplica(meta_previa):
        sal_nombre = (meta_previa.get("simulation", {}).get("salt") or {}).get("name", "?")
        print(f"  [NOTE] '{sal_nombre}' is a fully soluble salt: it does not "
              f"precipitate. This module does not apply.")
        return None

    prep = pc.preparar_comun(ruta_solucion, "2D PRECIPITATION MAP",
                             v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    if not pc.exigir_binaria(sol["meta"]):
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
            factor=FACTOR_PAPER, estilo="binario"):
    """Mapa limpio publication-ready (la leyenda se preserva: es la "escala").
    Nombres: binario {stem}_precipitation_{V}V_f{S}.png (el de v1.0.0),
    ternario {stem}_saturation_ternary_{V}V_f{S}.png, log10
    {stem}_saturation_log10_{V}V_f{S}.png. Batch: --opt factor=10 --opt estilo=ternary.
    con_datos=True → .txt con el conteo (S, Kps, nodos sobresaturados y que
    precipitan, max Q_act/Ksp)."""
    factor = float(factor)
    if not factor > 0:
        raise ValueError(f"factor (S) must be > 0, got {factor}")
    estilo = estilo_valido(estilo)
    sal = ctx["salt"]
    Kps = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2
    _, res = valores_mapa(datos, ctx, factor, estilo)
    stem, v = pathlib.Path(ruta_solucion).stem, ctx["v_label"]
    base = nombre_base(stem, v, factor, estilo)
    rutas = []
    if png:
        figc = crear_figura(datos, ctx, factor=factor, titulo=False,
                            figsize=(10, 4), estilo=estilo)
        rutas.append(pc.guardar_figura(figc, ruta_solucion, f"{base}.png", limpio=True))
    if con_datos:
        pct = 100.0 * res["n_precip"] / res["n_total"]
        fpath = (pathlib.Path(ruta_solucion).parent /
                 f"{nombre_base(stem, v, factor)}.txt")      # el mismo en los 3 estilos
        fpath.write_text(
            f"# Precipitation ({sal['name']}) | {v}V | {stem}\n"
            f"# Q_act = gamma±²·c+·c- (Davies) > factor·Kps\n"
            f"factor_x_Kps\t{factor:g}\n"
            f"Kps_M2\t{Kps:.6e}\n"
            f"nodos_precipitan\t{res['n_precip']}\n"
            f"nodos_totales\t{res['n_total']}\n"
            f"porcentaje\t{pct:.4f}\n"
            f"nodos_sobresaturados\t{res['n_sobresat']}\n"
            f"max_Qact_sobre_Kps\t{res['omega_max']:.6e}\n", encoding="utf-8")
        print(f"    ✓ Saved: {fpath}")
        rutas.append(fpath)
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo, TextBox de factor en vivo)
# =============================================================================
def _elegir_estilo():
    """Prompt del estilo del mapa (Enter = binario, el de siempre)."""
    print("    Map style: [1] binary (default)  [2] ternary (+ supersaturated "
          "below S)  [3] log10(Q_act/Ksp)")
    resp = input("    Style [Enter = 1]: ").strip().lower()
    opciones = {"": "binario", "1": "binario", "2": "ternario", "3": "log10"}
    if resp in opciones:
        return opciones[resp]
    try:
        return estilo_valido(resp)
    except ValueError:
        print("    (unknown style: using binary)")
        return "binario"


def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep
    estilo = _elegir_estilo()

    # ── Gráfico interactivo con TextBox ──────────────────────────────────────
    print(">>> Generating interactive graphical interface...")
    from matplotlib.widgets import TextBox

    factor_inicial = FACTOR_PAPER
    estado = {"factor": factor_inicial}

    fig = crear_figura(datos, ctx, factor=factor_inicial, estilo=estilo)
    ax  = fig.axes[0]
    cax = fig.axes[1] if estilo == "log10" else None   # barra fija de ±3 décadas
    fig.subplots_adjust(bottom=0.22)   # espacio inferior para el TextBox

    # Espejo precalculado UNA vez; cada redibujo solo recalcula los valores
    z2, r2, tri2, _ = pc.espejo(datos["z_nm"], datos["r_nm"], datos["tri"],
                                datos["cp_mM"])

    def refrescar_grafico(factor):
        valores, res = valores_mapa(datos, ctx, factor, estilo)
        ax.clear()
        dibujar_mapa(ax, z2, r2, tri2, np.concatenate([valores, valores]), ctx,
                     factor, info=res, estilo=estilo, colorbar=False)
        if cax is not None:            # mover la marca del umbral en la barra
            y = float(np.log10(factor))
            for linea in cax.lines:
                if linea.get_linestyle() == "--":
                    linea.set_ydata([y, y])
                    linea.set_visible(-LOG10_RANGO < y < LOG10_RANGO)
        fig.canvas.draw_idle()

    ax_box   = fig.add_axes([0.45, 0.06, 0.15, 0.045])
    text_box = TextBox(ax_box, "Threshold S (× Ksp): ", initial=f"{factor_inicial:g}")

    def on_submit(text_val):
        try:
            val = float(text_val)
            if val <= 0:
                print("⚠️ The supersaturation threshold S must be greater than 0.")
                return
            refrescar_grafico(val)
            estado["factor"] = val
            print(f">>> Map updated for S = {val:g}")
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
        guardar(datos, ctx, ruta_solucion, factor=estado["factor"], estilo=estilo)
    else:
        print("    (no image saved)")
