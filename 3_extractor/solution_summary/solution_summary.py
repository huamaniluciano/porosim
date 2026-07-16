# -*- coding: utf-8 -*-
"""
RESUMEN DE SOLUCIÓN — Pilar 3 (módulo especial).

Muestra un pantallazo ordenado de una solución abierta:
  - Texto en consola: geometría del canal, reservorios, subdivisiones de
    membrana, parámetros de la simulación y film(s).
  - Ventanas matplotlib: curva I-V y perfil del canal.

Contrato estándar del Pilar 3 (ver modulos/MODULE_CONTRACT.md):
    procesar(ruta_solucion)                    → consola (texto + ventanas)
    preparar(ruta, v_label=None, sol=None)     → (datos, ctx); NO usa voltaje
    guardar(datos, ctx, ruta, png, con_datos)  → PNGs I-V/esquema + reporte .txt
    USA_VOLTAJE = False                        → el batch no exige --voltaje

Lee el _sim.json (producido por el solver) y el IV_curve_*.txt.
SOLO LECTURA: no modifica el .h5 ni el JSON.

ROBUSTEZ: el _sim.json puede ser de una versión anterior al refactor
(formato viejo, sin los campos actuales). En ese caso el resumen NO se
rompe: avisa claro y muestra lo que pueda leer (modo degradado).

Ver README.md (Pilar 3) para el detalle de los módulos.
"""

import io
import json
import pathlib
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Módulo GLOBAL: analiza el barrido completo, no un voltaje puntual.
# El modo batch lo detecta y no exige --voltaje.
USA_VOLTAJE = False


# Paleta del esquema del canal (un solo lugar para tocar los colores).
PALETA = {
    "fluido":    "#eaf4fb",   # celeste muy pálido (electrolito)
    "film_pos":  "#f1948a",   # rosa/salmón (film de carga positiva)
    "film_neg":  "#7fb3d5",   # azul medio (film de carga negativa)
    "film_nd":   "#cacfd2",   # gris (film presente, signo desconocido)
    "membrana":  "#cacfd2",   # gris (membrana sólida)
    "carga_pos": "#cb4335",   # rojo (σ > 0)
    "carga_neg": "#1f618d",   # azul oscuro (σ < 0)
    "contorno":  "#34495e",   # gris azulado (contornos)
}


# Campos que SIEMPRE produce el mallador actual. Si falta alguno, el JSON
# es de un formato anterior al refactor (o está incompleto).
_MARCADORES_FORMATO_NUEVO = ("z_tip", "z_base", "R_tip", "R_base",
                             "L_pore", "L_res", "R_res")


# =============================================================================
# Utilidades de búsqueda y carga
# =============================================================================
def _buscar_sim_json(ruta_solucion):
    """Busca el _sim.json en la misma carpeta que el .h5. Devuelve dict o {}."""
    candidatos = list(ruta_solucion.parent.glob("*_sim.json"))
    if len(candidatos) == 1:
        with open(candidatos[0], "r", encoding="utf-8") as f:
            return json.load(f)
    elif len(candidatos) == 0:
        print("[NOTE] No _sim.json found in the folder. Very limited summary.")
        return {}
    else:
        print(f"[NOTE] Multiple _sim.json ({len(candidatos)}). Using: {candidatos[0].name}")
        with open(candidatos[0], "r", encoding="utf-8") as f:
            return json.load(f)


def _buscar_curva_iv(ruta_solucion):
    """Busca el IV_curve_*.txt en la misma carpeta. Devuelve Path o None."""
    candidatos = list(ruta_solucion.parent.glob("IV_curve_*.txt"))
    if candidatos:
        return candidatos[0]
    return None


def _diagnosticar_formato(meta):
    """
    Devuelve (es_compatible, campos_faltantes).
    Si faltan marcadores del formato nuevo, el JSON es viejo/incompleto.
    """
    if not meta:
        return False, list(_MARCADORES_FORMATO_NUEVO)
    faltan = [c for c in _MARCADORES_FORMATO_NUEVO if c not in meta]
    return (len(faltan) == 0), faltan


def _fmt_nm(valor_m):
    """Formatea un valor en metros como nm. None → 'n/d'."""
    if valor_m is None:
        return "n/a"
    return f"{valor_m * 1e9:.2f} nm"


# =============================================================================
# Bloques de texto en consola (todos defensivos: usan .get())
# =============================================================================
def _imprimir_geometria_canal(meta):
    print("\n=== CHANNEL GEOMETRY ===")
    R_tip  = meta.get("R_tip")
    R_base = meta.get("R_base")
    if meta.get("L_pore") is None and R_tip is None:
        print("  (no channel geometry data in the JSON)")
        return
    print(f"  Channel length   L_pore = {_fmt_nm(meta.get('L_pore'))}")
    # El mallador exporta RADIOS; el diámetro se deriva como 2·R.
    if R_tip is not None:
        print(f"  Tip mouth         R_tip  = {_fmt_nm(R_tip)}   "
              f"(D_tip  = {_fmt_nm(2 * R_tip)})")
    else:
        print(f"  Tip mouth         R_tip  = n/a")
    if R_base is not None:
        print(f"  Base mouth        R_base = {_fmt_nm(R_base)}   "
              f"(D_base = {_fmt_nm(2 * R_base)})")
    else:
        print(f"  Base mouth        R_base = n/a")
    print(f"  Tip mouth at z    z_tip  = {_fmt_nm(meta.get('z_tip'))}")
    print(f"  Base mouth at z   z_base = {_fmt_nm(meta.get('z_base'))}")
    # Tipo de perfil de pared. JSON viejos no lo traen → se asume cónico/lineal.
    canal_tipo = meta.get("channel_type", "conical")
    print(f"  Channel type       = {canal_tipo}")
    if canal_tipo == "bullet":
        print(f"  Bullet scale       h = {_fmt_nm(meta.get('h_param'))}")


def _imprimir_reservorios(meta):
    print("\n=== RESERVOIRS ===")
    if meta.get("L_res") is None and meta.get("R_res") is None:
        print("  (no reservoir data in the JSON)")
        return
    print(f"  Length of each reservoir  L_res = {_fmt_nm(meta.get('L_res'))}")
    print(f"  Reservoir radius          R_res = {_fmt_nm(meta.get('R_res'))}")


def _imprimir_membrana(meta):
    print("\n=== MEMBRANE SUBDIVISIONS ===")
    if meta.get("L_charge") is None and meta.get("L_far") is None:
        print("  (no subdivision data in the JSON)")
        return
    print(f"  Charged zone width   L_charge = {_fmt_nm(meta.get('L_charge'))}")
    print(f"  Fine buffer width    L_far    = {_fmt_nm(meta.get('L_far'))}")


def _imprimir_simulacion(meta):
    print("\n=== SIMULATION ===")
    sim = meta.get("simulation", {})
    if not sim:
        print("  (no 'simulation' section in the JSON)")
        return
    # Sal: bloque "salt" del catálogo; fallback legacy a campos planos sueltos.
    sal = sim.get("salt")
    if sal is not None:
        cat = sal.get("cation", {})
        an  = sal.get("anion", {})
        print(f"  Salt               = {sal.get('name', '?')}")
        if sal.get("soluble") is False:
            print(f"     partially soluble (Ksp = {sal.get('Ksp_M2', '?')} M²)")
        else:
            print(f"     fully soluble (does not precipitate)")
        z_p, z_m = cat.get("z", "?"), an.get("z", "?")
        D_p, D_m = cat.get("D_m2s", "?"), an.get("D_m2s", "?")
        sim_p, sim_m = cat.get("symbol", "cation"), an.get("symbol", "anion")
    else:
        # _sim.json legacy (anterior al catálogo de sales)
        print(f"  Electrolyte        = {sim.get('electrolito', '?')}")
        z_p, z_m = sim.get("z_p", "?"), sim.get("z_m", "?")
        D_p, D_m = sim.get("D_p_m2s", "?"), sim.get("D_m_m2s", "?")
        sim_p, sim_m = "cation", "anion"
    print(f"  Concentration c0   = {sim.get('c0_mM', '?')} mM")
    c_in  = sim.get("c0_inlet_mM")
    c_out = sim.get("c0_outlet_mM")
    if c_in is not None and c_out is not None and c_in != c_out:
        print(f"     (gradient: inlet={c_in} mM, outlet={c_out} mM)")
    sigma = sim.get("sigma_Cm2")
    if sigma is not None:
        print(f"  Surface charge     σ = {sigma:.4f} C/m²")
    print(f"  Temperature     T  = {sim.get('T_K', '?')} K")
    print(f"  Dielectric const εr = {sim.get('eps_r', '?')}")
    print(f"  Valences           z+ = {z_p} ({sim_p}), z- = {z_m} ({sim_m})")
    print(f"  Diffusivities      D+ = {D_p} m²/s, D- = {D_m} m²/s")
    print(f"  Maximum voltage    ±{sim.get('V_max_V', '?')} V "
          f"({sim.get('n_steps', '?')} puntos por rama)")


def _imprimir_films(meta):
    print("\n=== FILM(S) ===")
    sim = meta.get("simulation", {})
    films = sim.get("films", None)

    # Formato nuevo: lista 'films'.
    if isinstance(films, list):
        if not films:
            print("  Geometry with NO film.")
            return
        print(f"  {len(films)} active film(s):")
        for f in films:
            n_e = f.get("n_e_per_nm3")
            rho = f.get("rho_fix_target_Cm3")
            phi = f.get("phi_D_anal_mV")
            print(f"  • Film '{f.get('side', '?')}'  (type {f.get('type', '?')})")
            print(f"      n_e        = {n_e:.4f} e/nm³" if n_e is not None else "      n_e        = n/a")
            print(f"      ρ_fix      = {rho:.3e} C/m³" if rho is not None else "      ρ_fix      = n/a")
            print(f"      Φ_Donnan   = {phi:.2f} mV (analytic)" if phi is not None else "      Φ_Donnan   = n/a")
        return

    # Formato viejo: dict singular 'film' (pre-refactor).
    if isinstance(sim.get("film"), dict):
        print("  [OLD FORMAT] The JSON has a single 'film' (dict), not the")
        print("  current 'films' list. Showing what's available:")
        f = sim["film"]
        for k, v in f.items():
            print(f"      {k} = {v}")
        return

    print("  (no film information in the JSON)")


def _imprimir_mallado(meta, ruta_solucion):
    print("\n=== MESHING ===")
    # PENDIENTE: los parámetros de mallado
    # (lc_*, distancias, N_PTS_WALL) NO se exportan hoy al _sim.json.
    if "mesh_params" in meta:
        print("  (mesh design parameters)")
        for k, v in meta["mesh_params"].items():
            print(f"    {k} = {v}")
    else:
        print("  [PENDING] The mesh properties are not yet")
        print("  exported to the JSON.")


# =============================================================================
# Gráficos en ventanas matplotlib
# =============================================================================
def _texto_info_iv(meta):
    """
    Arma las líneas de info de la simulación para el recuadro de la curva I-V:
    electrolito, concentración y si el sistema es simétrico o no. Solo lee del
    _sim.json (defensivo con .get()). Devuelve [] si no hay nada que mostrar.
    """
    sim = meta.get("simulation", {}) if meta else {}
    if not sim:
        return []

    lineas = []

    # Electrolito: nombre de la sal (formato nuevo) o legacy 'electrolito'.
    sal = sim.get("salt")
    if isinstance(sal, dict):
        lineas.append(f"Electrolyte: {sal.get('name', '?')}")
    elif sim.get("electrolito") is not None:
        lineas.append(f"Electrolyte: {sim.get('electrolito')}")

    # Simétrico vs gradiente de concentración (inlet/outlet).
    c_in  = sim.get("c0_inlet_mM")
    c_out = sim.get("c0_outlet_mM")
    if c_in is not None and c_out is not None:
        if c_in == c_out:
            lineas.append(f"Symmetric: {c_in} mM")
        else:
            lineas.append("Asymmetric (gradient):")
            lineas.append(f"  inlet = {c_in} mM | outlet = {c_out} mM")
    elif sim.get("c0_mM") is not None:
        lineas.append(f"Concentration c0: {sim.get('c0_mM')} mM")

    return lineas


def _plot_curva_iv(ruta_iv, meta):
    """Grafica la curva I-V desde el archivo IV_curve_*.txt.
    Devuelve la Figure (o None si no se pudo graficar)."""
    if ruta_iv is None:
        print("\n[NOTE] No IV_curve_*.txt found; the I-V plot is omitted.")
        return None
    try:
        data = np.loadtxt(ruta_iv, skiprows=1)
    except Exception as e:
        print(f"\n[NOTE] Could not read {ruta_iv.name}: {e}")
        return None
    if data.ndim != 2 or data.shape[1] < 3:
        print(f"\n[NOTE] Unexpected format in {ruta_iv.name}; the plot is omitted.")
        return None

    V, I_in, I_out = data[:, 0], data[:, 1], data[:, 2]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(V, I_in,  'o-', label="I_in",  markersize=4)
    ax.plot(V, I_out, 's--', label="I_out", markersize=4)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [nA]")
    ax.set_title("I-V curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recuadro con la info de la simulación (electrolito / concentración / simetría).
    lineas = _texto_info_iv(meta)
    if lineas:
        ax.text(0.03, 0.97, "\n".join(lineas),
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#fcf3cf", alpha=0.9,
                          edgecolor="#b7950b"))

    fig.tight_layout()
    return fig


def _signo_film(film):
    """+1 si la carga del film es positiva, -1 si negativa, 0 si no se sabe."""
    v = film.get("n_e_per_nm3")
    if v is None:
        v = film.get("rho_fix_target_Cm3")
    if v is None:
        return 0
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _plot_perfil_canal(meta):
    """
    Esquema del canal en corte axisimétrico, ESPEJADO sobre el eje (r de -R a +R),
    con aspect equal. Rellena por región (fluido / film± / membrana) y colorea la
    carga superficial activa según su signo. Reconstruido solo desde el JSON.

    Soporta 0, 1 o 2 films (tip y/o base), simétricamente:
      - film tip  ocupa de z_film_tip  a z_tip   (a la IZQUIERDA de la boca tip)
      - film base ocupa de z_base       a z_film_base (a la DERECHA de la boca base)

    Soporta el perfil de pared lineal (cilindro/cónico) y el exponencial
    (bullet): para bullet se muestrea R(z) en vez de trazar una recta.
    """
    requeridos = ("L_pore", "R_tip", "R_base", "L_res", "R_res", "z_tip", "z_base")
    faltan = [c for c in requeridos if meta.get(c) is None]
    if faltan:
        print(f"\n[NOTE] Missing fields for the channel schematic ({', '.join(faltan)}); "
              f"the plot is omitted.")
        return None

    # --- Parámetros geométricos (en nm para el dibujo) ---
    Rtip  = meta["R_tip"]  * 1e9
    Rbase = meta["R_base"] * 1e9
    Rres  = meta["R_res"]  * 1e9
    ztip   = meta["z_tip"]   * 1e9
    zbase  = meta["z_base"]  * 1e9
    zin    = 0.0
    zout   = (2 * meta["L_res"] + meta["L_pore"]) * 1e9
    Lcharge = (meta.get("L_charge") or 0.0) * 1e9

    # --- Perfil de la pared (z, r) en nm, según el tipo de canal ---
    # Lineal (cilindro/cónico): basta con los dos extremos. Bullet: se muestrea
    # la exponencial R(z) = R_base - (R_base - R_tip)·exp(-(z - z_tip)/h).
    canal_tipo = meta.get("channel_type", "conical")
    if canal_tipo == "bullet" and meta.get("h_param"):
        h_nm = meta["h_param"] * 1e9
        zw = np.linspace(ztip, zbase, 80)
        rw = Rbase - (Rbase - Rtip) * np.exp(-(zw - ztip) / h_nm)
        rw[0] = Rtip      # ancla la boca tip exactamente en R_tip
        rw[-1] = Rbase    # ancla la boca base en R_base (igual que el spline del mallador)
    else:
        zw = np.array([ztip, zbase])
        rw = np.array([Rtip, Rbase])

    # --- Datos físicos (de la simulación) ---
    sim        = meta.get("simulation", {})
    tags       = meta.get("tags", {})
    tags_carga = sim.get("charge_tags", []) or []
    sigma      = sim.get("sigma_Cm2", 0.0) or 0.0
    films      = sim.get("films", []) or []

    # --- Film tip: presencia y extensión axial (IZQUIERDA de la boca tip) ---
    has_film_tip = bool(meta.get("include_film_tip")) and meta.get("z_film_tip") is not None
    zfilm_tip = meta["z_film_tip"] * 1e9 if has_film_tip else ztip
    film_tip  = next((f for f in films if f.get("side") == "tip"), None)

    # --- Film base: presencia y extensión axial (DERECHA de la boca base) ---
    has_film_base = bool(meta.get("include_film_base")) and meta.get("z_film_base") is not None
    zfilm_base = meta["z_film_base"] * 1e9 if has_film_base else zbase
    film_base  = next((f for f in films if f.get("side") == "base"), None)

    # Color de un film según el signo de su carga.
    def _col_film(film):
        if film is None:
            return PALETA["film_nd"]
        s = _signo_film(film)
        return (PALETA["film_pos"] if s > 0 else
                PALETA["film_neg"] if s < 0 else PALETA["film_nd"])

    fig, ax = plt.subplots(figsize=(11, 5))

    # =================== RELLENOS (cada uno simétrico en r) ===================
    # Reservorio tip (fluido): de inlet a la interfaz del film tip (o a la boca)
    ax.add_patch(mpatches.Rectangle((zin, -Rres), zfilm_tip - zin, 2 * Rres,
                 facecolor=PALETA["fluido"], edgecolor="none", zorder=1))
    # Film tip (color según signo de su carga)
    if has_film_tip:
        ax.add_patch(mpatches.Rectangle((zfilm_tip, -Rres), ztip - zfilm_tip, 2 * Rres,
                     facecolor=_col_film(film_tip), edgecolor="none", zorder=1))
    # Reservorio base (fluido): de la interfaz del film base (o de la boca) a outlet
    ax.add_patch(mpatches.Rectangle((zfilm_base, -Rres), zout - zfilm_base, 2 * Rres,
                 facecolor=PALETA["fluido"], edgecolor="none", zorder=1))
    # Film base (color según signo de su carga)
    if has_film_base:
        ax.add_patch(mpatches.Rectangle((zbase, -Rres), zfilm_base - zbase, 2 * Rres,
                     facecolor=_col_film(film_base), edgecolor="none", zorder=1))
    # Canal (fluido) — perfil de pared espejado sobre el eje
    ax.fill(np.r_[zw, zw[::-1]], np.r_[rw, -rw[::-1]],
            facecolor=PALETA["fluido"], edgecolor="none", zorder=1)
    # Membrana sólida (arriba y abajo de la pared del canal)
    ax.fill(np.r_[zw, zw[::-1]], np.r_[rw, np.full_like(rw, Rres)[::-1]],
            facecolor=PALETA["membrana"], edgecolor="none", zorder=1)
    ax.fill(np.r_[zw, zw[::-1]], np.r_[-rw, np.full_like(rw, -Rres)[::-1]],
            facecolor=PALETA["membrana"], edgecolor="none", zorder=1)

    # =================== CONTORNOS FINOS DEL DOMINIO ===================
    cc = PALETA["contorno"]
    ax.plot([zin, zout], [Rres, Rres],   color=cc, lw=0.8, zorder=2)
    ax.plot([zin, zout], [-Rres, -Rres], color=cc, lw=0.8, zorder=2)
    ax.plot([zin, zin],  [-Rres, Rres],  color=cc, lw=0.8, zorder=2)   # inlet
    ax.plot([zout, zout],[-Rres, Rres],  color=cc, lw=0.8, zorder=2)   # outlet
    # Caras de membrana (parte no cargada, trazo fino)
    ax.plot([ztip, ztip],   [Rtip, Rres],   color=cc, lw=0.8, zorder=2)
    ax.plot([ztip, ztip],   [-Rtip, -Rres], color=cc, lw=0.8, zorder=2)
    ax.plot([zbase, zbase], [Rbase, Rres],  color=cc, lw=0.8, zorder=2)
    ax.plot([zbase, zbase], [-Rbase, -Rres],color=cc, lw=0.8, zorder=2)
    ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=2)             # eje

    # =================== CARGA SUPERFICIAL (color por signo) ===================
    if sigma < 0:
        col_carga, etq_carga = PALETA["carga_neg"], "σ < 0"
    elif sigma > 0:
        col_carga, etq_carga = PALETA["carga_pos"], "σ > 0"
    else:
        col_carga, etq_carga = None, None

    def carga_activa(nombre):
        t = tags.get(nombre)
        return t is not None and t in tags_carga

    LW = 3.2
    # Pared del cono (WALL): coloreada si está cargada, si no trazo normal
    wall_on = carga_activa("WALL") and col_carga is not None
    wcol = col_carga if wall_on else cc
    wlw  = LW if wall_on else 1.0
    ax.plot(zw,  rw, color=wcol, lw=wlw, zorder=3)
    ax.plot(zw, -rw, color=wcol, lw=wlw, zorder=3)
    # Zona cargada de la cara tip
    if carga_activa("CHARGE_ZONE_TIP") and col_carga and Lcharge > 0:
        ax.plot([ztip, ztip], [Rtip, Rtip + Lcharge],   color=col_carga, lw=LW, zorder=4)
        ax.plot([ztip, ztip], [-Rtip, -(Rtip + Lcharge)],color=col_carga, lw=LW, zorder=4)
    # Zona cargada de la cara base
    if carga_activa("CHARGE_ZONE_BASE") and col_carga and Lcharge > 0:
        ax.plot([zbase, zbase], [Rbase, Rbase + Lcharge],   color=col_carga, lw=LW, zorder=4)
        ax.plot([zbase, zbase], [-Rbase, -(Rbase + Lcharge)],color=col_carga, lw=LW, zorder=4)

    # =================== MARCAS DE ESTACIONES ===================
    ax.axvline(ztip,  color="red",    ls="--", lw=0.6, alpha=0.4, zorder=2)
    ax.axvline(zbase, color="purple", ls="--", lw=0.6, alpha=0.4, zorder=2)
    if has_film_tip:
        ax.axvline(zfilm_tip,  color="orange", ls="--", lw=0.6, alpha=0.5, zorder=2)
    if has_film_base:
        ax.axvline(zfilm_base, color="green",  ls="--", lw=0.6, alpha=0.5, zorder=2)

    # =================== LEYENDA (dinámica) ===================
    handles = [mpatches.Patch(color=PALETA["fluido"], label="fluid (electrolyte)")]

    def _handle_film(film, lado):
        s = _signo_film(film) if film else 0
        signo_txt = "positivo" if s > 0 else "negativo" if s < 0 else "signo n/d"
        return mpatches.Patch(color=_col_film(film), label=f"{lado} film ({signo_txt})")

    if has_film_tip:
        handles.append(_handle_film(film_tip, "tip"))
    if has_film_base:
        handles.append(_handle_film(film_base, "base"))
    handles.append(mpatches.Patch(color=PALETA["membrana"], label="membrane (solid)"))
    if col_carga is not None and (wall_on or carga_activa("CHARGE_ZONE_TIP")
                                  or carga_activa("CHARGE_ZONE_BASE")):
        handles.append(Line2D([0], [0], color=col_carga, lw=3,
                              label=f"surface charge ({etq_carga})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    # =================== EJES ===================
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")
    ax.set_title("Channel schematic (axisymmetric cross-section, mirrored on the axis)")
    ax.set_xlim(zin, zout)
    ax.set_ylim(-Rres * 1.1, Rres * 1.1)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# =============================================================================
# Funciones públicas (contrato del Pilar 3)
# =============================================================================
def _imprimir_reporte(nombre_solucion, meta, ruta_solucion):
    """TODO el reporte de texto por print(): header + diagnóstico de formato
    + seccions. Lo usan procesar() (a consola) y guardar() (capturado con
    redirect_stdout hacia el .txt): una sola fuente del reporte."""
    print("\n" + "=" * 60)
    print(f"  SOLUTION SUMMARY: {nombre_solucion}")
    print("=" * 60)

    # Diagnóstico de formato: avisar claro si es viejo/incompleto.
    compatible, faltan = _diagnosticar_formato(meta)
    if meta and not compatible:
        print("\n" + "!" * 60)
        print("  [NOTE] This _sim.json seems to be from a version prior to the")
        print("  refactor, or is incomplete. Missing current-format fields:")
        print(f"    {', '.join(faltan)}")
        print("  The summary continues in DEGRADED MODE: it shows what it can read.")
        print("  For a complete summary, regenerate the solution with the mesher")
        print("  y el solver actuales.")
        print("!" * 60)

    _imprimir_geometria_canal(meta)
    _imprimir_reservorios(meta)
    _imprimir_membrana(meta)
    _imprimir_simulacion(meta)
    _imprimir_films(meta)
    _imprimir_mallado(meta, ruta_solucion)


def preparar(ruta_solucion, v_label=None, sol=None):
    """Contrato del extractor. Módulo GLOBAL: v_label y sol se ignoran (no
    necesita ni voltaje ni malla). datos = {meta, ruta_iv}; ctx mínimo."""
    ruta_solucion = pathlib.Path(ruta_solucion)
    return ({"meta":    _buscar_sim_json(ruta_solucion),
             "ruta_iv": _buscar_curva_iv(ruta_solucion)},
            {"stem": ruta_solucion.stem, "v_label": None})


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False):
    """PNG de la curva I-V + esquema del canal, y el reporte de texto como
    {stem}_summary.txt. El reporte se guarda SIEMPRE (es el output principal
    del módulo; con_datos se acepta por uniformidad del contrato)."""
    ruta_solucion = pathlib.Path(ruta_solucion)
    stem, carpeta = ruta_solucion.stem, ruta_solucion.parent
    rutas = []

    if png:
        for fig, nombre in ((_plot_curva_iv(datos["ruta_iv"], datos["meta"]),
                             f"{stem}_IV_curve.png"),
                            (_plot_perfil_canal(datos["meta"]),
                             f"{stem}_channel_schematic.png")):
            if fig is None:
                continue
            fpath = carpeta / nombre
            fig.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"    ✓ Saved: {fpath}")
            rutas.append(fpath)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _imprimir_reporte(ruta_solucion.name, datos["meta"], ruta_solucion)
    fpath = carpeta / f"{stem}_summary.txt"
    fpath.write_text(buf.getvalue().lstrip("\n") + "\n", encoding="utf-8")
    print(f"    ✓ Saved: {fpath}")
    rutas.append(fpath)
    return rutas


def procesar(ruta_solucion):
    ruta_solucion = pathlib.Path(ruta_solucion)
    datos, _ctx = preparar(ruta_solucion)

    # --- Texto en consola, por secciones ---
    _imprimir_reporte(ruta_solucion.name, datos["meta"], ruta_solucion)

    # --- Gráficos en ventanas aparte ---
    _plot_curva_iv(datos["ruta_iv"], datos["meta"])
    _plot_perfil_canal(datos["meta"])

    if plt.get_fignums():
        print("\n" + "=" * 60)
        print("  (close the plot windows to continue)")
        print("=" * 60)
        plt.show()   # bloquea hasta que se cierran las ventanas
    else:
        print("\n(no plots were generated)")