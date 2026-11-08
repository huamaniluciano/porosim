# -*- coding: utf-8 -*-
"""
Módulo: Precipitation onset vs voltage (GLOBAL, v1.1.0)
Categoría: precipitation_map

¿A qué voltaje aparece precipitado? Recorre TODOS los voltajes del .h5 (la
malla se carga una sola vez), se queda con max Ω(V) = max_dominio Q_act/Ksp y
grafica max Ω vs V en escala log, con el umbral S (precipita donde
Q_act > S·Ksp) y la línea de saturación Ω = 1. El voltaje de aparición de cada
polaridad es el primer |V| en que max Ω cruza S, interpolado linealmente entre
los voltajes calculados (precipitation.voltaje_umbral).

Misma física que precipitation.py (Davies con cap 0.5 M, mismo FACTOR_PAPER) y
mismo aplica(): solo sales parcialmente solubles con electrolito binario.

API pura (consola, batch y GUI):
    calcular_barrido(leer, labels, c0, Kps, z_p, z_m)  → datos
    crear_figura(datos, ctx, factor, v_read, ...)       → matplotlib Figure

datos = {"V": array [V], "S_max": array}   (ordenados por V)
ctx   = porosim_comun.contexto_de(meta, stem)   (sin voltaje: módulo global)

Batch (no usa --voltage):
    python launch_extractor.py <sol> precipitation_onset [--opt factor=10]
                               [--opt v_read=0.4] [--save-data]
→ {stem}_precipitation_onset_f{S}.png (+ .txt con la tabla V, max Ω).
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
_CAT = str(pathlib.Path(__file__).resolve().parent)
if _CAT not in sys.path:
    sys.path.insert(0, _CAT)
import porosim_comun as pc
import precipitation

# Módulo GLOBAL: analiza el barrido completo; el batch no exige --voltage.
USA_VOLTAJE = False

FACTOR_PAPER = precipitation.FACTOR_PAPER


def aplica(meta):
    """Lo mismo que el mapa de precipitación: sal poco soluble, binaria."""
    return precipitation.aplica(meta)


# =============================================================================
# FÍSICA (pura NumPy)
# =============================================================================
def calcular_barrido(leer_campos, labels, c0, Kps, z_p, z_m):
    """max Ω por voltaje. leer_campos(v_label) → campos de ese voltaje (el
    pc.leer_campos de la solución abierta). Solo se guarda el máximo: los
    campos completos de todo el barrido no hacen falta (ni entran en caché).
    → {"V": array ordenado, "S_max": array}."""
    V, smax = [], []
    for lab in sorted(labels, key=float):
        campos = leer_campos(lab)
        d = {"cp_mM": c0 * np.exp(campos["up"]), "cm_mM": c0 * np.exp(campos["um"])}
        V.append(float(lab))
        smax.append(precipitation.s_max(d, Kps, z_p, z_m))
    return {"V": np.array(V), "S_max": np.array(smax)}


def _ordenado(datos):
    V = np.asarray(datos["V"], dtype=float)
    S = np.asarray(datos["S_max"], dtype=float)
    orden = np.argsort(V)
    return V[orden], S[orden]


def _texto_onset(v_on):
    """{"+": 0.43, "-": None} → '+V: +0.43 V | -V: none in the sweep'."""
    return " | ".join(f"{s}V: " + ("none in the sweep" if v is None
                                   else f"{v if s == '+' else -v:+.3f} V")
                      for s, v in v_on.items())


# =============================================================================
# FIGURA (la misma para consola, batch y GUI)
# =============================================================================
def crear_figura(datos, ctx, factor=FACTOR_PAPER, v_read=None, leyenda=True,
                 titulo=True, figsize=(9, 5)):
    """max Q_act/Ksp vs V (semilog) con el umbral S, Ω = 1 y el voltaje de
    aparición de cada polaridad. v_read (opcional) sombrea |V| ≤ v_read (p. ej.
    el voltaje de lectura de un protocolo de pulsos). Deja el resultado en
    fig.onset = {"+": |V| o None, "-": |V| o None}."""
    V, smax = _ordenado(datos)
    v_on = precipitation.voltaje_umbral(V, smax, factor)
    sal = ctx["salt"]

    fig, ax = plt.subplots(figsize=figsize)
    if v_read:
        ax.axvspan(-abs(v_read), abs(v_read), color="0.9", zorder=0,
                   label=f"|V| ≤ V_read = {abs(v_read):g} V")
    ax.semilogy(V, smax, "o-", color="k", ms=3.5, lw=1.2,
                label="max Q_act/Ksp in the domain")
    ax.axhline(factor, color="#FF0000", ls="--", lw=1.2,
               label=f"threshold S = {factor:g} (precipitates above)")
    ax.axhline(1.0, color="0.45", ls=":", lw=0.9, label="saturation (Q_act = Ksp)")
    for signo, v in v_on.items():
        if v is None:
            continue
        vv = v if signo == "+" else -v
        ax.axvline(vv, color="#FF0000", lw=0.8, alpha=0.6)
        ax.annotate(f"onset {vv:+.2f} V", (vv, factor), textcoords="offset points",
                    xytext=(5 if signo == "+" else -5, 7), fontsize=8,
                    color="#FF0000", ha="left" if signo == "+" else "right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.85))
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Applied voltage V [V]")
    ax.set_ylabel("max Q_act / Ksp")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    if leyenda:
        ax.legend(fontsize=8, loc="best", framealpha=0.95)
    if titulo:
        c0 = ctx.get("c0")
        ax.set_title(f"Precipitation onset ({sal['name']}"
                     + (f", {c0:g} mM" if c0 is not None else "")
                     + f") — {ctx['stem']}\n{_texto_onset(v_on)}", fontsize=10)
    fig.tight_layout()
    fig.onset = v_on
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """Módulo global: ignora v_label. Lee todos los voltajes (reusa `sol` si
    viene cargada) → ({"V", "S_max"}, ctx) o None si no aplica / falla."""
    ruta = pathlib.Path(ruta_solucion)
    print("\n" + "=" * 60)
    print("   MÓDULO: PRECIPITATION ONSET VS VOLTAGE")
    print("=" * 60)
    meta = sol["meta"] if sol is not None else pc.cargar_meta(ruta.parent)
    if not meta:
        print("❌ _sim.json not found in the solution folder.")
        return None
    if not pc.aplica_binaria(meta):
        pc.exigir_binaria(meta)                 # prints why (mixture / no c0)
        return None
    if not aplica(meta):
        nombre = (meta.get("simulation", {}).get("salt") or {}).get("name", "?")
        print(f"  [NOTE] '{nombre}' is a fully soluble salt: it does not "
              f"precipitate. This module does not apply.")
        return None
    if sol is None:
        print(f">>> Loading mesh from: {ruta.name}")
        try:
            sol = pc.cargar_solucion(ruta)
        except Exception as e:
            print(f"❌ Error loading the mesh/solution: {e}")
            return None
    if not sol["labels"]:
        print("❌ No solution datasets found in the file.")
        return None

    ctx = pc.contexto_de(sol["meta"], ruta.stem)
    sal = ctx["salt"]
    Kps = sal.get("Ksp_M2") or pc.KPS_LEGACY_M2
    print(f"  Electrolyte: {pc.describir_electrolito(sol['meta'].get('simulation', {}))}"
          f" | Ksp = {Kps:.2e} M²")
    print(f">>> Reading {len(sol['labels'])} voltages (max Q_act/Ksp of each) ...")
    try:
        datos = calcular_barrido(
            lambda lab: pc.leer_campos(sol["ruta"], sol["mesh"], sol["V"], lab),
            sol["labels"], ctx["c0"], Kps, sal["z_p"], sal["z_m"])
    except Exception as e:
        print(f"❌ Error reading the sweep: {e}")
        return None
    return datos, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False,
            factor=FACTOR_PAPER, v_read=None):
    """Figura completa (con ejes: es un gráfico 1D) y, con con_datos, la tabla
    V, max Ω. Nombres: {stem}_precipitation_onset_f{S}.png / .txt.
    Batch: --opt factor=10 --opt v_read=0.4."""
    factor = float(factor)
    if not factor > 0:
        raise ValueError(f"factor (S) must be > 0, got {factor}")
    v_read = abs(float(v_read)) if v_read else None
    stem = pathlib.Path(ruta_solucion).stem
    base = f"{stem}_precipitation_onset_f{factor:g}"
    V, smax = _ordenado(datos)
    v_on = precipitation.voltaje_umbral(V, smax, factor)
    print(f"  Onset at S = {factor:g}: {_texto_onset(v_on)}")
    rutas = []
    if png:
        fig = crear_figura(datos, ctx, factor=factor, v_read=v_read)
        rutas.append(pc.guardar_figura(fig, ruta_solucion, f"{base}.png"))
    if con_datos:
        sal = ctx["salt"]
        rutas.append(pc.exportar_txt(
            ruta_solucion, f"{base}.txt", [V, smax],
            f"max Q_act/Ksp vs V (Davies, cap 0.5 M) | {sal['name']} | {stem}\n"
            f"onset at S = {factor:g}: {_texto_onset(v_on)}\n"
            f"V_V\tmax_Qact_over_Ksp"))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep
    V, smax = _ordenado(datos)
    print("\n      V [V]    max Q_act/Ksp")
    for v, s in zip(V, smax):
        print(f"    {v:+7.2f}    {s:.4g}")

    txt = input(f"\nThreshold S (× Ksp) [Enter = {FACTOR_PAPER:g}]: ").strip()
    try:
        factor = float(txt) if txt else FACTOR_PAPER
        if not factor > 0:
            raise ValueError
    except ValueError:
        print(f"    (invalid S: using {FACTOR_PAPER:g})")
        factor = FACTOR_PAPER

    crear_figura(datos, ctx, factor=factor)
    print(">>> Opening interactive window. Close it to return to the menu.")
    plt.show()

    resp = input("\nSave the figure and the V, max Q_act/Ksp table? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion, con_datos=True, factor=factor)
    else:
        print("    (nothing saved)")
