# -*- coding: utf-8 -*-
"""
Módulo: I-V metrics (GLOBAL, v1.1.0)
Categoría: iv_analysis

Métricas de forma de la curva I-V del barrido (tabla IV_curve_*.txt del
solver; no necesita FEniCS ni el .h5). Son las del análisis de rectificación y
forma S del paper de precipitación (respuesta al referato):

    f_rec    = |I(+V_ref) / I(−V_ref)|                  rectificación
    G0       = pendiente de I(V) en |V| ≤ V_G0 [nS]    conductancia lineal
    n±       = d ln|I| / d ln|V| ajustada por mínimos cuadrados en la
               ventana de |V| (0.7–1 V), en cada rama: localmente I ∝ |V|ⁿ.
               n > 1 superlineal (rectificante), n = 1 óhmico, n < 1
               sublineal (saturación por polarización: forma S)
    S-index± = G(V_ref) / max G(0 < |V| ≤ V_ref), con G = I/V, en cada rama.
               Vale 1 sin forma S; menos de 1 si la conductancia cae a
               voltaje alto
    pendiente local d ln|I| / d ln|V| vs |V| (figura): dónde aparece la S

I = promedio de la corriente del inlet y del outlet (las dos columnas de la
tabla), como en el análisis del paper: su diferencia es el error de
conservación del solver. Los voltajes de referencia se toman del punto más
cercano de la tabla (se informa cuál se usó).

API pura (consola, batch y GUI):
    cargar_iv(ruta_txt)                     → datos {"V", "I_in", "I_out", "I"}
    metricas(V, I, ventana_n, v_ref, v_g0)  → dict
    pendiente_local(V, I)                   → por rama, (|V|, d ln|I|/d ln|V|)
    crear_figura(datos, ctx, ...)           → Figure (I-V + pendiente local)

Batch (no usa --voltage):
    python launch_extractor.py <sol> iv_metrics [--opt v_ref=0.5]
                               [--opt ventana_n=0.3,0.5] [--save-data]
→ {stem}_IV_metrics.txt (siempre) + .png (+ la pendiente local con --save-data).
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc

# Módulo GLOBAL: analiza la tabla I-V del barrido; el batch no exige --voltage.
USA_VOLTAJE = False

VENTANA_N = (0.7, 1.0)    # V: ventana del ajuste de n (la del análisis del paper)
V_REF     = 1.0           # V: f_rec y S-index
V_G0      = 0.1           # V: G0 = pendiente de I(V) en |V| ≤ V_G0
_TOL_V    = 1e-9


# =============================================================================
# LECTURA
# =============================================================================
def buscar_iv(carpeta):
    """IV_curve_*.txt de la carpeta de la solución (o el curva_IV*.txt de
    corridas viejas). None si no hay."""
    carpeta = pathlib.Path(carpeta)
    for patron in ("IV_curve_*.txt", "curva_IV*.txt"):
        candidatos = sorted(carpeta.glob(patron))
        if candidatos:
            return candidatos[0]
    return None


def cargar_iv(ruta_txt):
    """Tabla del solver (V, I_in, I_out [nA], una fila de encabezado) →
    {"V", "I_in", "I_out", "I"} ordenada por V, con I = (I_in + I_out)/2."""
    d = np.atleast_2d(np.loadtxt(ruta_txt, skiprows=1))
    if d.shape[1] < 3:
        raise ValueError(f"{pathlib.Path(ruta_txt).name}: expected the columns "
                         f"V, I_in, I_out (got {d.shape[1]} columns)")
    d = d[np.argsort(d[:, 0])]
    return {"V": d[:, 0], "I_in": d[:, 1], "I_out": d[:, 2],
            "I": 0.5 * (d[:, 1] + d[:, 2])}


# =============================================================================
# MÉTRICAS (pura NumPy)
# =============================================================================
def _rama(V, I, signo):
    """(|V|, |I|) de una rama (signo = +1 o −1), sin V = 0 y solo donde la
    corriente tiene el signo del voltaje (si no, ln|I| no describe la rama),
    ordenada por |V|."""
    V, I = np.asarray(V, dtype=float), np.asarray(I, dtype=float)
    m = (signo * V > _TOL_V) & (signo * I > 0)
    v, i = np.abs(V[m]), np.abs(I[m])
    orden = np.argsort(v)
    return v[orden], i[orden]


def valor_en(V, I, v):
    """(V usado, I) en el voltaje de la tabla más cercano a v."""
    V = np.asarray(V, dtype=float)
    k = int(np.argmin(np.abs(V - v)))
    return float(V[k]), float(np.asarray(I, dtype=float)[k])


def exponente_n(V, I, ventana=VENTANA_N, signo=+1):
    """n = pendiente de ln|I| vs ln|V| (mínimos cuadrados) en la ventana de
    |V| de la rama `signo`. None si la ventana tiene menos de 2 puntos."""
    v, i = _rama(V, I, signo)
    m = (v >= ventana[0] - _TOL_V) & (v <= ventana[1] + _TOL_V)
    if np.count_nonzero(m) < 2:
        return None
    return float(np.polyfit(np.log(v[m]), np.log(i[m]), 1)[0])


def s_index(V, I, v_ref=V_REF, signo=+1):
    """G(v_ref)/max G(0 < |V| ≤ v_ref) en la rama `signo`, G = I/V (la
    referencia es el |V| de la rama más cercano a v_ref). None sin datos."""
    v, i = _rama(V, I, signo)
    if len(v) == 0:
        return None
    k = int(np.argmin(np.abs(v - v_ref)))
    G = i[:k + 1] / v[:k + 1]
    return float(G[-1] / G.max())


def conductancia_g0(V, I, v_g0=V_G0):
    """Pendiente de I(V) [nS si I en nA] por mínimos cuadrados en |V| ≤ v_g0.
    None si hay menos de 2 puntos."""
    V, I = np.asarray(V, dtype=float), np.asarray(I, dtype=float)
    m = np.abs(V) <= v_g0 + _TOL_V
    if np.count_nonzero(m) < 2:
        return None
    return float(np.polyfit(V[m], I[m], 1)[0])


def factor_rectificacion(V, I, v_ref=V_REF):
    """(|I(+v)/I(−v)|, +v usado, −v usado). None si falta una de las ramas o
    I(−v) = 0."""
    V = np.asarray(V, dtype=float)
    if not (np.any(V > _TOL_V) and np.any(V < -_TOL_V)):
        return None, None, None
    vp, ip = valor_en(V[V > _TOL_V], np.asarray(I)[V > _TOL_V], v_ref)
    vm, im = valor_en(V[V < -_TOL_V], np.asarray(I)[V < -_TOL_V], -v_ref)
    if im == 0:
        return None, vp, vm
    return abs(ip / im), vp, vm


def pendiente_local(V, I):
    """Pendiente local d ln|I| / d ln|V| de cada rama (diferencias finitas
    de segundo orden sobre ln-ln) → {"+": (|V|, n_local), "-": (...)}."""
    out = {}
    for clave, signo in (("+", +1), ("-", -1)):
        v, i = _rama(V, I, signo)
        n_loc = (np.gradient(np.log(i), np.log(v)) if len(v) >= 2
                 else np.full(len(v), np.nan))
        out[clave] = (v, n_loc)
    return out


def metricas(V, I, ventana_n=VENTANA_N, v_ref=V_REF, v_g0=V_G0):
    """Todas las métricas de una curva I-V (V en V, I en nA). Las que no se
    pueden calcular (rama ausente, ventana sin puntos) quedan en None."""
    ventana_n = tuple(float(x) for x in np.atleast_1d(ventana_n))
    if len(ventana_n) != 2 or not 0 < ventana_n[0] < ventana_n[1]:
        raise ValueError(f"ventana_n must be two voltages 0 < a < b, got {ventana_n}")
    v_ref, v_g0 = float(v_ref), float(v_g0)
    if not v_ref > 0 or not v_g0 > 0:
        raise ValueError(f"v_ref and v_g0 must be > 0 (got {v_ref}, {v_g0})")
    f_rec, vp, vm = factor_rectificacion(V, I, v_ref)
    I_p = valor_en(V, I, vp)[1] if vp is not None else None
    I_m = valor_en(V, I, vm)[1] if vm is not None else None
    return {"f_rec": f_rec, "v_ref_pos": vp, "v_ref_neg": vm,
            "I_ref_pos_nA": I_p, "I_ref_neg_nA": I_m,
            "G0_nS": conductancia_g0(V, I, v_g0),
            "n_pos": exponente_n(V, I, ventana_n, +1),
            "n_neg": exponente_n(V, I, ventana_n, -1),
            "S_index_pos": s_index(V, I, v_ref, +1),
            "S_index_neg": s_index(V, I, v_ref, -1),
            "ventana_n": ventana_n, "v_ref": v_ref, "v_g0": v_g0}


def texto_metricas(m):
    """Las métricas como líneas 'clave<TAB>valor' (reporte, .txt y consola)."""
    def f(x, fmt):
        return "n/a" if x is None else format(x, fmt)
    a, b = m["ventana_n"]
    return [f"f_rec\t{f(m['f_rec'], '.4g')}\t|I({f(m['v_ref_pos'], '+.2f')} V) / "
            f"I({f(m['v_ref_neg'], '+.2f')} V)|",
            f"I_ref_pos_nA\t{f(m['I_ref_pos_nA'], '.6g')}",
            f"I_ref_neg_nA\t{f(m['I_ref_neg_nA'], '.6g')}",
            f"G0_nS\t{f(m['G0_nS'], '.6g')}\tslope of I(V) for |V| <= {m['v_g0']:g} V",
            f"n_pos\t{f(m['n_pos'], '.4f')}\td ln I / d ln V fitted on {a:g}-{b:g} V (+V branch)",
            f"n_neg\t{f(m['n_neg'], '.4f')}\tsame on the -V branch (|I| vs |V|)",
            f"S_index_pos\t{f(m['S_index_pos'], '.4f')}\tG(V_ref)/max G(0 < V <= V_ref), G = I/V",
            f"S_index_neg\t{f(m['S_index_neg'], '.4f')}\tsame on the -V branch"]


# =============================================================================
# FIGURA (la misma para consola, batch y GUI)
# =============================================================================
def crear_figura(datos, ctx, ventana_n=VENTANA_N, v_ref=V_REF, v_g0=V_G0,
                 leyenda=True, titulo=True, figsize=(12, 4.8)):
    """(a) I-V con G0 y los voltajes de referencia; (b) pendiente local
    d ln|I|/d ln|V| de cada rama, con n = 1 (óhmico) y la ventana del ajuste
    de n. Deja las métricas en fig.metricas."""
    V, I = datos["V"], datos["I"]
    m = metricas(V, I, ventana_n, v_ref, v_g0)
    fig, (ax_iv, ax_n) = plt.subplots(1, 2, figsize=figsize)

    ax_iv.plot(V, I, "o-", color="#2980b9", ms=3.5, lw=1.4, label="I = (I_in + I_out)/2")
    if m["G0_nS"] is not None:
        vv = np.array([V.min(), V.max()]) * 0.35
        ax_iv.plot(vv, m["G0_nS"] * vv, "--", color="0.45", lw=1.0,
                   label=f"G0 = {m['G0_nS']:.3g} nS (|V| ≤ {m['v_g0']:g} V)")
    for v in (m["v_ref_pos"], m["v_ref_neg"]):
        if v is not None:
            ax_iv.axvline(v, color="#e67e22", lw=0.8, ls=":")
    ax_iv.axhline(0, color="gray", lw=0.6)
    ax_iv.axvline(0, color="gray", lw=0.6)
    ax_iv.set_xlabel("Applied voltage V [V]")
    ax_iv.set_ylabel("Current I [nA]")
    ax_iv.grid(True, ls=":", alpha=0.3)
    if m["f_rec"] is not None:
        ax_iv.text(0.03, 0.97, f"f_rec = {m['f_rec']:.3g}  (±{abs(m['v_ref_pos']):g} V)",
                   transform=ax_iv.transAxes, va="top", fontsize=9,
                   bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    a, b = m["ventana_n"]
    ax_n.axvspan(a, b, color="0.92", zorder=0, label=f"fit window of n ({a:g}–{b:g} V)")
    ax_n.axhline(1.0, color="0.4", lw=0.9, ls=":", label="n = 1 (ohmic)")
    for clave, color, marca in (("+", "#c0392b", "o-"), ("-", "#2471a3", "s--")):
        v, n_loc = pendiente_local(V, I)[clave]
        if len(v):
            n_fit = m["n_pos" if clave == "+" else "n_neg"]
            etiqueta = f"{clave}V branch" + (f"  (n = {n_fit:.3f})" if n_fit is not None else "")
            ax_n.plot(v, n_loc, marca, color=color, ms=3.5, lw=1.2, label=etiqueta)
    ax_n.set_xlabel("|V| [V]")
    ax_n.set_ylabel("local slope d ln|I| / d ln|V|")
    ax_n.grid(True, ls=":", alpha=0.3)
    if leyenda:
        ax_iv.legend(fontsize=8, loc="lower right")
        ax_n.legend(fontsize=8, loc="best")
    if titulo:
        fig.suptitle(f"I-V metrics — {ctx.get('stem', '')}", fontsize=11)
    fig.tight_layout()
    fig.metricas = m
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    """Módulo global: ignora v_label y sol (no lee el .h5).
    → ({"V", "I_in", "I_out", "I"}, ctx) o None si no hay tabla I-V."""
    ruta = pathlib.Path(ruta_solucion)
    print("\n" + "=" * 60)
    print("   MÓDULO: I-V METRICS")
    print("=" * 60)
    ruta_iv = buscar_iv(ruta.parent)
    if ruta_iv is None:
        print("❌ No IV_curve_*.txt in the solution folder (the solver writes it "
              "next to the .h5).")
        return None
    try:
        datos = cargar_iv(ruta_iv)
    except Exception as e:
        print(f"❌ Could not read {ruta_iv.name}: {e}")
        return None
    print(f"  I-V table: {ruta_iv.name} ({len(datos['V'])} points)")
    meta = sol["meta"] if sol is not None else pc.cargar_meta(ruta.parent)
    return datos, pc.contexto_de(meta or {}, ruta.stem)


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False,
            ventana_n=VENTANA_N, v_ref=V_REF, v_g0=V_G0):
    """{stem}_IV_metrics.txt SIEMPRE (es la salida principal), la figura con
    png, y con con_datos la pendiente local de cada rama
    ({stem}_IV_local_slope.txt). Batch: --opt ventana_n=0.3,0.5 --opt v_ref=0.5."""
    m = metricas(datos["V"], datos["I"], ventana_n, v_ref, v_g0)
    stem = pathlib.Path(ruta_solucion).stem
    carpeta = pathlib.Path(ruta_solucion).parent
    lineas = texto_metricas(m)
    for linea in lineas:
        print("  " + linea.replace("\t", "  "))
    rutas = []
    fpath = carpeta / f"{stem}_IV_metrics.txt"
    fpath.write_text(f"# I-V metrics | {stem}\n"
                     f"# I = (I_in + I_out)/2 [nA]; reference voltages are the closest "
                     f"table points\n" + "\n".join(lineas) + "\n", encoding="utf-8")
    print(f"    ✓ Saved: {fpath}")
    rutas.append(fpath)
    if png:
        fig = crear_figura(datos, ctx, ventana_n, v_ref, v_g0)
        rutas.append(pc.guardar_figura(fig, ruta_solucion, f"{stem}_IV_metrics.png"))
    if con_datos:
        pend = pendiente_local(datos["V"], datos["I"])
        filas = [(s * v, n) for clave, s in (("-", -1), ("+", +1))
                 for v, n in zip(*pend[clave])]
        filas.sort()
        if filas:
            rutas.append(pc.exportar_txt(
                ruta_solucion, f"{stem}_IV_local_slope.txt",
                [np.array([f[0] for f in filas]), np.array([f[1] for f in filas])],
                f"local slope d ln|I| / d ln|V| of each branch | {stem}\nV_V\tlocal_slope"))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep
    m = metricas(datos["V"], datos["I"])
    print()
    for linea in texto_metricas(m):
        print("  " + linea.replace("\t", "  "))
    crear_figura(datos, ctx)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()
    resp = input("\nSave the metrics (.txt) and the figure? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion, con_datos=True)
    else:
        print("    (nothing saved)")
