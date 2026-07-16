# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — SOLVER · GUI: MOTOR DE DIBUJO FÍSICO
  Dibuja el canal con un overlay cromático que representa la física:
    - Azul = Carga negativa (pared o film catiónico)
    - Rojo = Carga positiva (pared o film aniónico)
    - Intensidad del color = Magnitud de la carga
  Dibuja también el "Proto-Plot" de los puntos de voltaje a muestrear.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Añadir la carpeta del mallador al path para usar la fórmula de perfil único
AQUI = os.path.dirname(os.path.abspath(__file__))
MALLADOR_DIR = os.path.join(os.path.dirname(AQUI), "1_mesher")
sys.path.insert(0, MALLADOR_DIR)
from capa1_modelo import perfil_radio

NM = 1e9   # m → nm

# Paleta Física
PALETA = {
    "fluido":     "#f4f8fb",   # electrolito neutro
    "membrana":   "#e0e0e0",   # membrana sólida
    "contorno":   "#2c3e50",   # contornos del dominio
    "eje":        "#7f8c8d",   # línea del eje
    
    # Colores físicos (divergentes)
    "negativo":   "#2980b9",   # azul
    "positivo":   "#c0392b",   # rojo
    "neutro":     "#7f8c8d",   # gris
}


def _R_perfil(st, z, z_tip, z_base):
    """Llama a perfil_radio del mallador para que los perfiles coincidan exacto."""
    return perfil_radio(st.get("type"), st["D_tip"]/2.0, st["D_base"]/2.0,
                        z, z_tip, z_base - z_tip, st.get("h_param"))


def _col_magnitud(valor, tipo_carga="superficial"):
    """Devuelve el color hex y el alfa según el signo y la magnitud."""
    if valor == 0.0:
        return PALETA["neutro"], 0.2
    
    color = PALETA["positivo"] if valor > 0 else PALETA["negativo"]
    
    # Calcular alfa proporcional a la magnitud
    val_abs = abs(valor)
    if tipo_carga == "superficial":
        # Mapea de 0 a 2.0 e/nm²
        alpha = 0.2 + 0.8 * min(val_abs / 2.0, 1.0)
    else:
        # Molar del film: mapea de 0 a 4.0 M
        alpha = 0.15 + 0.7 * min(val_abs / 4.0, 1.0)
        
    return color, alpha


def dibujar_canal_fisica(ax, st, sigma_e_nm2, usar_corona, films_cfg):
    """
    Dibuja el canal con coloración física (azul=negativo, rojo=positivo).
    films_cfg: lista de dicts de films activos en esta corrida, ej:
               [{"name": "tip", "molar": 4.0, "signo": -1}, ...]
    """
    ax.clear()

    R_tip, R_base = st["D_tip"]/2.0, st["D_base"]/2.0
    L_pore = st["L_pore"]
    
    hay_res = bool(st.get("L_res") and st.get("R_res"))
    L_res = st["L_res"] if hay_res else None
    R_res = st["R_res"] if hay_res else None

    # Alto de referencia del dibujo
    Rdib = R_res if hay_res else max(R_base, R_tip) + st.get("L_charge", 0.0) + 6e-9
    
    z_tip  = L_res if hay_res else (0.30 * L_pore)
    z_base = z_tip + L_pore
    if hay_res:
        z_in, z_out = 0.0, 2 * L_res + L_pore
    else:
        margen = 0.30 * L_pore
        z_in, z_out = z_tip - margen, z_base + margen

    # Buscar si hay films y su configuración
    ft_cfg = next((f for f in films_cfg if f["side"] == "tip"), None)
    fb_cfg = next((f for f in films_cfg if f["side"] == "base"), None)

    delta_tip = st["film_tip"]["delta"] if st.get("film_tip") else 0.0
    delta_base = st["film_base"]["delta"] if st.get("film_base") else 0.0
    
    z_film_tip  = z_tip  - delta_tip
    z_film_base = z_base + delta_base

    # 1. Relleno de fluido
    def rect(z0, z1, color, alpha=1.0):
        ax.add_patch(mpatches.Rectangle((z0*NM, -Rdib*NM), (z1-z0)*NM, 2*Rdib*NM,
                     facecolor=color, alpha=alpha, edgecolor="none", zorder=1))

    # Fluido neutral en reservorios
    rect(z_in, z_film_tip, PALETA["fluido"])
    rect(z_film_base, z_out, PALETA["fluido"])
    
    # Relleno del film tip
    if ft_cfg and delta_tip > 0:
        val_carga = ft_cfg["molar"] * ft_cfg["signo"]
        col, alp = _col_magnitud(val_carga, "volumetrica")
        rect(z_film_tip, z_tip, col, alp)
    elif delta_tip > 0:
        # Film existente en la malla pero sin carga definida
        rect(z_film_tip, z_tip, PALETA["neutro"], 0.2)
        
    # Relleno del film base
    if fb_cfg and delta_base > 0:
        val_carga = fb_cfg["molar"] * fb_cfg["signo"]
        col, alp = _col_magnitud(val_carga, "volumetrica")
        rect(z_base, z_film_base, col, alp)
    elif delta_base > 0:
        rect(z_base, z_film_base, PALETA["neutro"], 0.2)

    # Fluido en el canal
    if st.get("type") == "bullet":
        u = np.linspace(0.0, 1.0, 240)
        zs = z_tip + u * u * (z_base - z_tip)
    else:
        zs = np.linspace(z_tip, z_base, 240)
    rs = np.array([_R_perfil(st, z, z_tip, z_base) for z in zs])
    rs[0], rs[-1] = R_tip, R_base
    
    ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[rs, -rs[::-1]]*NM,
            facecolor=PALETA["fluido"], edgecolor="none", zorder=1)

    # 2. Relleno de Membrana Sólida
    ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[rs, np.full_like(rs, Rdib)[::-1]]*NM,
            facecolor=PALETA["membrana"], edgecolor="none", zorder=1)
    ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[-rs, np.full_like(rs, -Rdib)[::-1]]*NM,
            facecolor=PALETA["membrana"], edgecolor="none", zorder=1)

    # 3. Contornos estructurales
    cc = PALETA["contorno"]
    ax.plot([z_tip*NM]*2,  [R_tip*NM, Rdib*NM],   color=cc, lw=0.8, zorder=2)
    ax.plot([z_tip*NM]*2,  [-R_tip*NM, -Rdib*NM], color=cc, lw=0.8, zorder=2)
    ax.plot([z_base*NM]*2, [R_base*NM, Rdib*NM],  color=cc, lw=0.8, zorder=2)
    ax.plot([z_base*NM]*2, [-R_base*NM, -Rdib*NM],color=cc, lw=0.8, zorder=2)
    ax.axhline(0, color=PALETA["eje"], lw=0.6, ls=":", zorder=2)

    # 4. Coloración física de la pared cargada (sigma)
    col_sigma, alp_sigma = _col_magnitud(sigma_e_nm2, "superficial")
    
    # Trazo de la pared cargada
    ax.plot(zs*NM,  rs*NM, color=col_sigma, alpha=alp_sigma, lw=3.0, zorder=4)
    ax.plot(zs*NM, -rs*NM, color=col_sigma, alpha=alp_sigma, lw=3.0, zorder=4)
    # Contorno fino de la pared
    ax.plot(zs*NM,  rs*NM, color=cc, lw=0.6, zorder=5)
    ax.plot(zs*NM, -rs*NM, color=cc, lw=0.6, zorder=5)

    # 5. Coloración de las coronas de membrana (si aplica)
    if usar_corona and st.get("L_charge", 0) > 0:
        Lc = st["L_charge"]
        Lf = st.get("L_far", 0.0)
        # Si sigma_en_coronas es True, pintamos del color de la carga
        col_c = col_sigma
        alp_c = alp_sigma
        
        for (zf, Rb) in [(z_tip, R_tip), (z_base, R_base)]:
            # zona cargada
            ax.plot([zf*NM]*2, [Rb*NM, (Rb+Lc)*NM],   color=col_c, alpha=alp_c, lw=4.0, zorder=4)
            ax.plot([zf*NM]*2, [-Rb*NM, -(Rb+Lc)*NM], color=col_c, alpha=alp_c, lw=4.0, zorder=4)
            # Bordes finos
            ax.plot([zf*NM]*2, [Rb*NM, (Rb+Lc)*NM],   color=cc, lw=0.6, zorder=5)
            ax.plot([zf*NM]*2, [-Rb*NM, -(Rb+Lc)*NM], color=cc, lw=0.6, zorder=5)
            
            # zona de transición (punteada fina)
            if Lf > 0:
                ax.plot([zf*NM]*2, [(Rb+Lc)*NM, (Rb+Lc+Lf)*NM], color=col_c, lw=1.2, ls=":", zorder=5)
                ax.plot([zf*NM]*2, [-(Rb+Lc)*NM, -(Rb+Lc+Lf)*NM], color=col_c, lw=1.2, ls=":", zorder=5)

    # Marcas de guía axiales
    ax.axvline(z_tip*NM,  color="red",    ls="--", lw=0.5, alpha=0.3, zorder=2)
    ax.axvline(z_base*NM, color="purple", ls="--", lw=0.5, alpha=0.3, zorder=2)
    if delta_tip > 0: ax.axvline(z_film_tip*NM,  color="blue", ls="--", lw=0.5, alpha=0.3, zorder=2)
    if delta_base > 0: ax.axvline(z_film_base*NM, color="green", ls="--", lw=0.5, alpha=0.3, zorder=2)

    # Configuración de ejes
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.2)
    
    # Zoom ajustado al canal + films + un pequeño margen
    margin = 0.25 * L_pore
    r_zoom = max(R_base, R_tip) + st.get("L_charge", 0.0)
    ax.set_xlim((z_film_tip - margin)*NM, (z_film_base + margin)*NM)
    ax.set_ylim(-r_zoom*1.6*NM, r_zoom*1.6*NM)
    ax.set_title("Physical Domain Visualization")


def leyenda_handles_fisica(sigma_e_nm2, films_cfg):
    """Genera las leyendas dinámicas según los signos de las cargas."""
    h = [mpatches.Patch(color=PALETA["fluido"], label="fluid (electrolyte)"),
         mpatches.Patch(color=PALETA["membrana"], label="membrane (solid)")]

    # Pared
    if sigma_e_nm2 < 0:
        h.append(Line2D([0], [0], color=PALETA["negativo"], lw=3, label=f"charged wall (negative: {sigma_e_nm2:g} e/nm²)"))
    elif sigma_e_nm2 > 0:
        h.append(Line2D([0], [0], color=PALETA["positivo"], lw=3, label=f"charged wall (positive: {sigma_e_nm2:g} e/nm²)"))
    else:
        h.append(Line2D([0], [0], color=PALETA["neutro"], lw=3, label="neutral wall"))

    # Films
    for f in films_cfg:
        lado = f["side"]
        molar = f["molar"]
        signo = f["signo"]
        t = "cationic/negative" if signo < 0 else "anionic/positive"
        col = PALETA["negativo"] if signo < 0 else PALETA["positivo"]
        h.append(mpatches.Patch(color=col, alpha=0.4, label=f"{lado} film ({t}: {molar:g}M)"))
        
    return h


def dibujar_proto_plot(ax, V_max, n_steps):
    """
    Dibuja el Proto-Plot de la curva I-V esperada.
    Representa los voltajes a resolver sobre un plano I-V vacío.
    """
    ax.clear()
    
    # Generar puntos de voltaje
    v_pos = np.linspace(0.0, V_max, n_steps)
    v_neg = np.linspace(0.0, -V_max, n_steps)
    voltajes = np.unique(np.concatenate([v_neg, v_pos]))
    
    # Eje X de voltajes, Eje Y de corriente vacío
    ax.axhline(0, color="gray", lw=0.8, ls="-")
    ax.axvline(0, color="gray", lw=0.8, ls="-")
    
    # Dibujar las pautas de muestreo
    for v in voltajes:
        ax.axvline(v, color="#f39c12", ls=":", lw=0.8, alpha=0.6)
        
    ax.scatter(voltajes, np.zeros_like(voltajes), color="#d35400", s=40, zorder=3,
               label=f"Voltage points ({len(voltajes)} total)")
    
    ax.set_xlim(-V_max * 1.1, V_max * 1.1)
    ax.set_ylim(-1, 1)  # vacío
    ax.get_yaxis().set_ticks([]) # ocultar ticks de Y ya que no hay valores resueltos aún
    
    ax.set_xlabel("Applied voltage V [V]")
    ax.set_ylabel("Current I [nA] (expected)")
    ax.set_title("Proto-Plot: I-V curve sampling")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(loc="upper left")
