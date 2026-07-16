# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MALLADOR · GUI: MOTOR DE DIBUJO (renderiza el estado de diseño)
  Dibuja un Params en el estilo de resumen-solucion (fills por región).
  Soporta: cilindro (D_tip==D_base) / cónico, films tip/base, y el caso
  AÚN SIN RESERVORIOS (margen neutro a los lados, claramente no-reservorio).
═══════════════════════════════════════════════════════════════════════════

  Filosofía: el dibujo NO inventa geometría. Toma un Params y lo dibuja usando
  el MISMO perfil (perfil_radio, la fórmula única de la Capa 1) que usan las
  capas: el dibujo y la malla no pueden divergir. El cilindro no es un caso
  aparte: es el cónico con D_tip == D_base (la GUI mostrará un solo diámetro,
  pero el Params sigue llevando los dos).

  Estado del Params según la etapa de diseño:
    - Reservorios NO definidos  → params.L_res / params.R_res == None.
      Se dibuja el canal+films "al aire", con un margen neutro a los costados
      de las bocas (placeholder visual, gris muy claro rayado) que deja claro
      que ahí todavía no hay nada definido.
    - Reservorios definidos     → se dibujan como fluido a ambos lados y la
      vista completa muestra la proporción real.

  Requiere capa1_modelo.py en la misma carpeta (solo para perfil_radio).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from capa1_modelo import perfil_radio


NM = 1e9   # m → nm

# Paleta NEUTRA / GEOMÉTRICA. En el dibujador NO existe la carga (eso es del
# solver): los colores describen geometría, no física. Los films se distinguen
# por LADO (tip / base), no por signo.
PALETA = {
    "fluido":     "#eaf4fb",   # electrolito
    "film":       "#c8b6a6",   # film (neutro: sin polaridad, sin distinguir lado)
    "membrana":   "#cacfd2",   # membrana sólida
    "contorno":   "#34495e",   # contornos del dominio
    "pared":      "#2c3e50",   # pared del canal (resaltada, geométrica)
    "corona":     "#5d6d7e",   # zona de carga en la cara (geométrica, neutra)
    "sin_def":    "#f4f6f7",   # placeholder: zona aún no definida (reservorios)
    "sin_def_ln": "#bdc3c7",   # borde rayado del placeholder
}


# =================================================================
# DESCRIPCIÓN DEL ESTADO DE DISEÑO (un dict liviano, NO el Params final)
# =================================================================
# Mientras se diseña, el "params" puede estar incompleto (sin reservorios).
# Usamos un dict simple con las claves que el dibujo necesita; la conversión a
# un Params real (para generar) se hace recién cuando están los reservorios.
#
#   D_tip, D_base   [m]   diámetros de boca (iguales = cilindro)
#   L_pore          [m]   largo del canal
#   tipo            str   "cylinder" | "conical" | (futuro: "bullet", ...)
#   film_tip, film_base   dict o None  → {"delta":[m], "signo":+1/-1/0}
#   L_res, R_res    [m] o None  → None = reservorios aún no definidos
#   sigma_signo     +1/-1/0     (solo para colorear la carga superficial)
#   L_charge        [m]   ancho de la zona cargada de la cara (para el dibujo)
# =================================================================
def estado_demo(**kw):
    """Crea un estado de diseño con defaults razonables, para pruebas."""
    base = dict(
        type="cylinder",
        D_tip=10e-9, D_base=10e-9, L_pore=100e-9,
        film_tip=None, film_base=None,
        L_res=None, R_res=None,
        L_charge=5e-9,
        h_param=None,   # escala del bullet [m]; None salvo tipo=="bullet"
    )
    base.update(kw)
    # cilindro ⇒ forzar D_base = D_tip (la GUI lo hará; acá lo garantizamos)
    if base["type"] == "cylinder":
        base["D_base"] = base["D_tip"]
    return base


# =================================================================
# HELPERS
# =================================================================
def _R_tip(st):  return st["D_tip"] / 2.0
def _R_base(st): return st["D_base"] / 2.0


def _R_perfil(st, z, z_tip, z_base):
    """Radio de la pared a la altura z. Llama a perfil_radio() de la Capa 1
    (la FÓRMULA ÚNICA del perfil) con los valores del estado de la GUI, sin
    construir un Params completo. Así el dibujo y la malla no pueden divergir."""
    return perfil_radio(st.get("type"), _R_tip(st), _R_base(st),
                        z, z_tip, z_base - z_tip, st.get("h_param"))


def _col_film(lado=None):
    """Color del film: ÚNICO y neutro. En el dibujador no hay polaridad ni se
    distingue tip/base por color (la posición y la etiqueta ya los distinguen)."""
    return PALETA["film"]


def _R_dibujo(st):
    """Radio vertical de referencia para el dibujo. Si hay reservorios, R_res;
    si no, un alto cómodo basado en la boca mayor + la cara de membrana."""
    if st.get("R_res"):
        return st["R_res"]
    return max(_R_base(st), _R_tip(st)) + st.get("L_charge", 0.0) + 6e-9


# =================================================================
# DIBUJO
# =================================================================
def dibujar_canal(ax, st, vista="completa"):
    """
    Dibuja el estado de diseño `st` en el eje `ax`.
      vista="completa" → toda la pieza (con reservorios si existen)
      vista="zoom"     → acercado al canal + films

    No crea figura ni llama a show(): solo dibuja en `ax` (apto para refrescar
    en vivo desde una GUI).
    """
    ax.clear()

    R_tip, R_base = _R_tip(st), _R_base(st)
    Rdib = _R_dibujo(st)
    L_pore = st["L_pore"]

    hay_res = bool(st.get("L_res") and st.get("R_res"))
    L_res = st["L_res"] if hay_res else None
    R_res = st["R_res"] if hay_res else None

    # --- Marco axial: ubicamos el canal y derivamos las posiciones ---
    # z_tip = boca tip; z_base = boca base. Si hay reservorios, se extienden a
    # los lados; si no, ponemos un margen neutro corto (placeholder).
    z_tip  = L_res if hay_res else (0.30 * L_pore)   # margen neutro = 30% del largo
    z_base = z_tip + L_pore
    if hay_res:
        z_in, z_out = 0.0, 2 * L_res + L_pore
    else:
        margen = 0.30 * L_pore
        z_in, z_out = z_tip - margen, z_base + margen

    # --- Films: extensión axial ---
    ft, fb = st.get("film_tip"), st.get("film_base")
    z_film_tip  = z_tip  - ft["delta"] if ft else z_tip
    z_film_base = z_base + fb["delta"] if fb else z_base

    # ========================= RELLENOS =========================
    def rect(z0, z1, color, **kw):
        ax.add_patch(mpatches.Rectangle((z0*NM, -Rdib*NM), (z1-z0)*NM, 2*Rdib*NM,
                     facecolor=color, edgecolor="none", zorder=1, **kw))

    # Zona izquierda (reservorio fluido, o placeholder "sin definir")
    if hay_res:
        rect(z_in, z_film_tip, PALETA["fluido"])
    else:
        ax.add_patch(mpatches.Rectangle((z_in*NM, -Rdib*NM), (z_film_tip-z_in)*NM, 2*Rdib*NM,
                     facecolor=PALETA["sin_def"], edgecolor=PALETA["sin_def_ln"],
                     hatch="////", lw=0.0, zorder=1))
    # Film tip
    if ft:
        rect(z_film_tip, z_tip, _col_film("tip"))
    # Zona derecha (reservorio fluido, o placeholder)
    if hay_res:
        rect(z_film_base, z_out, PALETA["fluido"])
    else:
        ax.add_patch(mpatches.Rectangle((z_film_base*NM, -Rdib*NM), (z_out-z_film_base)*NM, 2*Rdib*NM,
                     facecolor=PALETA["sin_def"], edgecolor=PALETA["sin_def_ln"],
                     hatch="////", lw=0.0, zorder=1))
    # Film base
    if fb:
        rect(z_base, z_film_base, _col_film("base"))

    # Cono/cilindro/bullet (fluido) — perfil espejado. Muestreo adaptativo para
    # bullet: concentra puntos al tip donde la exponencial cambia rápido.
    # Para recta (cilindro/cónico): linspace uniforme (es suficiente).
    if st.get("type") == "bullet":
        u = np.linspace(0.0, 1.0, 240)
        zs = z_tip + u * u * (z_base - z_tip)
    else:
        zs = np.linspace(z_tip, z_base, 240)
    rs = np.array([_R_perfil(st, z, z_tip, z_base) for z in zs])
    # Anclar las bocas exactamente en R_tip/R_base, igual que el spline del
    # mallador (Capa 3): para bullet la fórmula solo tiende a R_base, así que
    # sin esto quedaría un hueco contra la cara de membrana dibujada en R_base.
    rs[0], rs[-1] = R_tip, R_base
    ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[rs, -rs[::-1]]*NM,
            facecolor=PALETA["fluido"], edgecolor="none", zorder=1)
    # Membrana (arriba y abajo de la pared del canal)
    ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[rs, np.full_like(rs, Rdib)[::-1]]*NM,
            facecolor=PALETA["membrana"], edgecolor="none", zorder=1)
    ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[-rs, np.full_like(rs, -Rdib)[::-1]]*NM,
            facecolor=PALETA["membrana"], edgecolor="none", zorder=1)

    # ========================= CONTORNOS =========================
    cc = PALETA["contorno"]
    # pared del canal (espejada)
    ax.plot(zs*NM,  rs*NM, color=cc, lw=1.4, zorder=3)
    ax.plot(zs*NM, -rs*NM, color=cc, lw=1.4, zorder=3)
    # caras de membrana (verticales en las bocas)
    ax.plot([z_tip*NM]*2,  [R_tip*NM, Rdib*NM],   color=cc, lw=0.8, zorder=2)
    ax.plot([z_tip*NM]*2,  [-R_tip*NM, -Rdib*NM], color=cc, lw=0.8, zorder=2)
    ax.plot([z_base*NM]*2, [R_base*NM, Rdib*NM],  color=cc, lw=0.8, zorder=2)
    ax.plot([z_base*NM]*2, [-R_base*NM, -Rdib*NM],color=cc, lw=0.8, zorder=2)
    ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=2)

    # ========================= PARED DEL CANAL (geométrica) =========================
    # En el dibujador NO hay carga: la pared se resalta solo como límite
    # geométrico del canal (la carga se define después, en el solver).
    ax.plot(zs*NM,  rs*NM, color=PALETA["pared"], lw=2.0, zorder=4)
    ax.plot(zs*NM, -rs*NM, color=PALETA["pared"], lw=2.0, zorder=4)

    # ========================= CORONA (subdivisión de la cara) =========================
    # Segmento de la cara de membrana marcado como zona de carga (geométrico,
    # neutro: sin color de polaridad). Simétrico tip/base. Ancho L_charge desde
    # la boca; un guion fino marca el fin de L_far (transición).
    if st.get("usar_corona") and st.get("L_charge", 0) > 0:
        Lc = st["L_charge"]; Lf = st.get("L_far", 0.0)
        col = PALETA["corona"]
        for (zf, Rb) in [(z_tip, R_tip), (z_base, R_base)]:
            # zona cargada (trazo grueso neutro)
            ax.plot([zf*NM]*2, [Rb*NM, (Rb+Lc)*NM],   color=col, lw=3.0, zorder=5)
            ax.plot([zf*NM]*2, [-Rb*NM, -(Rb+Lc)*NM], color=col, lw=3.0, zorder=5)
            # fin de L_far (marca fina punteada)
            if Lf > 0:
                ax.plot([zf*NM]*2, [(Rb+Lc)*NM, (Rb+Lc+Lf)*NM],
                        color=col, lw=1.2, ls=":", zorder=5)
                ax.plot([zf*NM]*2, [-(Rb+Lc)*NM, -(Rb+Lc+Lf)*NM],
                        color=col, lw=1.2, ls=":", zorder=5)

    # ========================= MARCAS DE ESTACIÓN =========================
    ax.axvline(z_tip*NM,  color="red",    ls="--", lw=0.6, alpha=0.4, zorder=2)
    ax.axvline(z_base*NM, color="purple", ls="--", lw=0.6, alpha=0.4, zorder=2)
    if ft: ax.axvline(z_film_tip*NM,  color="orange", ls="--", lw=0.6, alpha=0.5, zorder=2)
    if fb: ax.axvline(z_film_base*NM, color="green",  ls="--", lw=0.6, alpha=0.5, zorder=2)

    # ========================= LÍMITES Y ETIQUETAS =========================
    ax.set_xlabel("z [nm]"); ax.set_ylabel("r [nm]")
    # adjustable="datalim": el recuadro del plot NO cambia de forma al variar la
    # geometría (lo fija el figsize); matplotlib expande el rango de datos para
    # mantener la escala 1:1 (z y r con la misma unidad), agregando solo márgenes.
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.25)

    if vista == "zoom":
        margin = 0.3 * L_pore
        r_zoom = max(R_base, R_tip) + st.get("L_charge", 0.0)
        ax.set_xlim((z_film_tip - margin)*NM, (z_film_base + margin)*NM)
        ax.set_ylim(-r_zoom*1.6*NM, r_zoom*1.6*NM)
        ax.set_title("channel / film(s) zoom")
    else:
        ax.set_xlim(z_in*NM, z_out*NM)
        ax.set_ylim(-Rdib*1.15*NM, Rdib*1.15*NM)
        tt = "full view" if hay_res else "channel view (no reservoirs yet)"
        ax.set_title(tt)


def leyenda_handles(st):
    """Handles de leyenda según lo presente en el estado."""
    h = [mpatches.Patch(color=PALETA["fluido"], label="fluid (electrolyte)")]
    if st.get("film_tip") or st.get("film_base"):
        h.append(mpatches.Patch(color=_col_film(), label="film"))
    h.append(mpatches.Patch(color=PALETA["membrana"], label="membrane (solid)"))
    h.append(Line2D([0], [0], color=PALETA["pared"], lw=2, label="channel wall"))
    if st.get("usar_corona") and st.get("L_charge", 0) > 0:
        h.append(Line2D([0], [0], color=PALETA["corona"], lw=3,
                        label="charge zone (geometric)"))
    if not (st.get("L_res") and st.get("R_res")):
        h.append(mpatches.Patch(facecolor=PALETA["sin_def"], edgecolor=PALETA["sin_def_ln"],
                                hatch="////", label="undefined reservoir"))
    return h


# =================================================================
# PRUEBA (genera PNGs de varios estados)
# =================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    casos = [
        ("cilindro_sin_res",
         estado_demo(type="cylinder", D_tip=10e-9, L_pore=100e-9)),
        ("conico_sin_res",
         estado_demo(type="conical", D_tip=10e-9, D_base=50e-9, L_pore=100e-9)),
        ("conico_film_tip_sin_res",
         estado_demo(type="conical", D_tip=10e-9, D_base=50e-9, L_pore=100e-9,
                     film_tip={"delta": 12e-9})),
        ("conico_dos_films_sin_res",
         estado_demo(type="conical", D_tip=10e-9, D_base=50e-9, L_pore=100e-9,
                     film_tip={"delta": 12e-9},
                     film_base={"delta": 12e-9})),
        ("conico_dos_films_con_res",
         estado_demo(type="conical", D_tip=10e-9, D_base=50e-9, L_pore=100e-9,
                     film_tip={"delta": 12e-9},
                     film_base={"delta": 12e-9},
                     L_res=500e-9, R_res=400e-9)),
    ]

    for nombre, st in casos:
        fig, (axz, axc) = plt.subplots(2, 1, figsize=(12, 9))
        dibujar_canal(axz, st, vista="zoom")
        dibujar_canal(axc, st, vista="completa")
        axc.legend(handles=leyenda_handles(st), loc="upper right", fontsize=8, framealpha=0.9)
        fig.suptitle(nombre, fontsize=12)
        fig.tight_layout()
        fig.savefig(f"dibujo_{nombre}.png", dpi=110)
        plt.close(fig)
        print(f"  ✓ dibujo_{nombre}.png")
    print("Listo.")