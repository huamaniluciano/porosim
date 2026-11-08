# -*- coding: utf-8 -*-
"""
Módulo: Mapa de Potential 2D
Categoría: potential_map

Genera un mapa de color 2D del potencial eléctrico phi para un voltaje
elegido por el usuario, usando tricontourf (igual que ion_maps).

Funciona con o sin films. Cada film se dibuja como un borde:
    rojo  si el film tiene carga fija positiva
    azul  si el film tiene carga fija negativa
El interior del film NO se rellena (ahí va la info del potencial).

API pura (compartida entre consola, batch y GUI):
    crear_figura(datos, ctx, con_campo=False, ...)   → matplotlib Figure

Contrato del extractor (ver modulos/MODULE_CONTRACT.md):
    preparar(ruta, v_label=None, sol=None)  → (datos, ctx) | None
    guardar(datos, ctx, ruta, png, con_datos) → [Paths]
    procesar(ruta)                           → cáscara de consola

datos = {"z_nm", "r_nm", "tri", "phi_V"} (arrays SIN espejar, φ en Volts);
ctx   = porosim_comun.contexto_de(meta, stem, v_label).

con_campo=True superpone las líneas de campo E = -∇φ (ver campo_electrico()
en porosim_comun-nivel; acá se calcula con el mismo CubicTriInterpolator).

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

_MOD = str(pathlib.Path(__file__).resolve().parents[1])
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
import porosim_comun as pc


# =============================================================================
# CAMPO ELÉCTRICO E = -∇φ (pura NumPy/matplotlib.tri) — compartido con field_lines
# =============================================================================
def campo_electrico(z_nm, r_nm, tri, phi_V, zlim=None, rlim=None, n_r=55, n_z=None):
    """E en grilla regular + espejo axisimétrico (Ez par, Er impar), con
    CubicTriInterpolator directo sobre la malla (sin project()). Devuelve
    {Z, R_full, Ez, Er} listo para streamplot (NaN fuera del dominio).

    zlim/rlim (nm): acotan la grilla a una ventana (zoom). La grilla tiene un
    número FIJO de nodos: sobre todo el dominio el paso es de ~90 nm y una
    ventana de 100 nm no llega a contener un solo nodo, así que sin esto el
    zoom sale sin líneas de campo. La ventana se recorta al dominio (no se
    extrapola) y, como la grilla se arma en r ≥ 0 y después se espeja, de rlim
    se toma |r|max.

    n_r / n_z: nodos de la grilla en r ≥ 0 y en z (n_z=None → según el aspecto
    de la ventana)."""
    z_min, z_max = z_nm.min(), z_nm.max()
    r_min, r_max = r_nm.min(), r_nm.max()
    if zlim is not None:                      # recortado al dominio: no extrapolar
        z_min, z_max = max(z_min, min(zlim)), min(z_max, max(zlim))
    if rlim is not None:
        r_max = min(r_max, max(abs(rlim[0]), abs(rlim[1])))
        r_min = 0.0
    aspect = (z_max - z_min) / (r_max - r_min) if (r_max - r_min) > 0 else 1.0

    # Construir el interpolador sobre la malla ENTERA cuesta ~10 s en una malla
    # de 650k nodos. Con una ventana se recorta la malla a su vecindad (con un
    # margen para que el gradiente del borde siga bien apoyado): ~90x más
    # rápido y el mismo resultado.
    z_sub, r_sub, tri_sub, phi_sub = z_nm, r_nm, tri, phi_V
    if zlim is not None or rlim is not None:
        pad_z = 0.5 * (z_max - z_min)
        pad_r = 0.5 * (r_max - r_min)
        cerca = ((z_nm >= z_min - pad_z) & (z_nm <= z_max + pad_z) &
                 (r_nm >= r_min - pad_r) & (r_nm <= r_max + pad_r))
        # solo triángulos con sus 3 vértices adentro (si no, índices colgados)
        keep = cerca[tri].all(axis=1)
        if keep.sum() >= 16:                  # si queda muy poco, usar todo
            idx = np.flatnonzero(cerca)
            remap = np.full(len(z_nm), -1, dtype=np.int64)
            remap[idx] = np.arange(len(idx))
            z_sub, r_sub, phi_sub = z_nm[idx], r_nm[idx], phi_V[idx]
            tri_sub = remap[tri[keep]]

    triang = mtri.Triangulation(z_sub, r_sub, tri_sub)
    if n_z is None:
        n_z = int(np.clip(n_r * aspect, 80, 400))
    Z = np.linspace(z_min, z_max, n_z)
    R = np.linspace(r_min, r_max, n_r)
    Z_grid, R_grid = np.meshgrid(Z, R)

    interp = mtri.CubicTriInterpolator(triang, phi_sub, kind="geom")
    dphi_dz, dphi_dr = interp.gradient(Z_grid, R_grid)
    Ez = np.ma.filled(-dphi_dz, np.nan)   # NaN = fuera del dominio (streamplot)
    Er = np.ma.filled(-dphi_dr, np.nan)

    # Sobre el eje (r = 0) la axisimetría impone ∂φ/∂r = 0, pero el
    # interpolador hace un ajuste local sobre una malla no estructurada y no lo
    # sabe: devuelve un Er espurio (~25 % del campo típico), justo donde pasan
    # las líneas centrales. Se impone la condición de contorno; np.where
    # preserva los NaN (donde el eje cae fuera del fluido no se inventa campo).
    if np.isclose(R[0], 0.0):
        Er[0, :] = np.where(np.isfinite(Er[0, :]), 0.0, np.nan)

    R_neg, Ez_neg, Er_neg = -R[::-1], Ez[::-1, :], -Er[::-1, :]
    if np.isclose(R_neg[-1], R[0]):       # evitar fila duplicada en r = 0
        R_neg, Ez_neg, Er_neg = R_neg[:-1], Ez_neg[:-1, :], Er_neg[:-1, :]
    return {"Z": Z, "R_full": np.concatenate([R_neg, R]),
            "Ez": np.vstack([Ez_neg, Ez]), "Er": np.vstack([Er_neg, Er])}


def _punto_medio(v):
    """(punto, tangente unitaria) a mitad del largo de arco de una polilínea."""
    d = np.hypot(np.diff(v[:, 0]), np.diff(v[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return None
    i = int(np.clip(np.searchsorted(s, s[-1] / 2.0), 1, len(v) - 1))
    dz, dr = v[i, 0] - v[i - 1, 0], v[i, 1] - v[i - 1, 1]
    n = np.hypot(dz, dr)
    if n == 0:
        return None
    return v[i], (dz / n, dr / n)


def dibujar_lineas_campo(ax, Z, R, Ez, Er, density=0.7, color="0.55",
                         linewidth=0.8, arrowsize=1.0):
    """Líneas de campo con simetría axial EXACTA.

    streamplot es codicioso: recorre su malla interna en un orden fijo y marca
    las celdas que ocupa, así que aun con un campo perfectamente simétrico
    devuelve trayectorias distintas arriba y abajo (medido: 20 contra 17, ~6 %
    de diferencia de largo). Acá se integra SOLO en r ≥ 0 (con Er = 0 en el eje
    ninguna línea puede cruzarlo, ver campo_electrico) y cada trayectoria se
    dibuja además reflejada. Rinde ~el doble de líneas por unidad de density
    que un streamplot sobre la ventana entera.

    Las flechas se ponen a mano en el medio de cada trayectoria: las de
    streamplot son patches en coordenadas de PANTALLA y no se pueden espejar.
    Cada flecha es un patch, así que en el dominio completo (cientos de líneas)
    esto es lento: crear_figura lo usa solo en los zooms, donde la asimetría
    se ve.
    """
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection

    sel = R >= 0
    figtmp, axtmp = plt.subplots()
    try:
        sp = axtmp.streamplot(Z, R[sel], Ez[sel], Er[sel], density=density)
        caminos = [p.vertices.copy() for p in sp.lines.get_paths()]
    finally:
        plt.close(figtmp)

    segmentos = []
    for v in caminos:
        segmentos.append(v)
        if np.abs(v[:, 1]).max() > 1e-9:      # la línea del eje no se duplica
            espejada = v.copy()
            espejada[:, 1] *= -1.0
            segmentos.append(espejada)

    ax.add_collection(LineCollection(segmentos, colors=color,
                                     linewidths=linewidth, zorder=2),
                      autolim=False)          # no tocar los límites ya fijados
    paso = 1e-3 * np.hypot(Z[-1] - Z[0], R[-1] - R[0])
    for v in segmentos:
        medio = _punto_medio(v)
        if medio is None:
            continue
        (z0, r0), (dz, dr) = medio
        ax.add_patch(mpatches.FancyArrowPatch(
            (z0 - dz * paso, r0 - dr * paso), (z0 + dz * paso, r0 + dr * paso),
            arrowstyle="->", mutation_scale=10 * arrowsize,
            color=color, linewidth=linewidth, zorder=3))
    return len(segmentos)


# =============================================================================
# FIGURA (la misma para consola y GUI). con_campo → líneas de campo encima.
# =============================================================================
def crear_figura(datos, ctx, con_campo=False, leyenda=True,
                 ylim=None, xlim=None, titulo=True, figsize=None,
                 campo=None):
    """campo: dict {Z, R_full, Ez, Er} ya calculado (p.ej. cacheado por la
    GUI). Si es None y con_campo=True, se calcula acá con campo_electrico()."""
    z_nm, r_nm, tri, phi_V = (datos["z_nm"], datos["r_nm"],
                              datos["tri"], datos["phi_V"])
    z2, r2, tri2, phi2 = pc.espejo(z_nm, r_nm, tri, phi_V)
    norm, niveles = pc.escala_potencial(phi_V, ctx.get("V_T"))
    es_zoom = (xlim is not None) or (ylim is not None)

    # Con zoom se ignora el campo cacheado (grilla de todo el dominio, ~90 nm
    # de paso) y se recalcula sobre la ventana: si no, no hay nodos que dibujar.
    ec = campo
    if con_campo and (ec is None or es_zoom):
        ec = campo_electrico(z_nm, r_nm, tri, phi_V,
                             zlim=xlim if es_zoom else None,
                             rlim=ylim if es_zoom else None)

    if figsize is None:
        figsize = (14, 6) if con_campo else (14, 5)
    fig, ax = plt.subplots(figsize=figsize)
    # extend="neither": escala_potencial ya arma niveles que cubren todo el
    # dato, así que un triángulo de extend no representaría nada, y además se
    # pinta con el color extremo del cmap sin pasar por el norm (salto de color
    # falso al pie de la barra).
    im = ax.tricontourf(z2, r2, tri2, phi2, levels=niveles,
                        cmap="RdBu_r", norm=norm, extend="neither")

    # Límites explícitos: sin zoom, los del dato YA espejado (el autoscale de
    # matplotlib agrega un margen y muestra dominio donde no lo hay).
    ax.set_aspect("equal")
    ax.set_ylim(*(ylim if ylim is not None else (r2.min(), r2.max())))
    ax.set_xlim(*(xlim if xlim is not None else (z2.min(), z2.max())))

    if ec is not None:
        if es_zoom:
            # En el zoom se ve la asimetría de streamplot: integrar media
            # ventana y espejar (ver dibujar_lineas_campo).
            dibujar_lineas_campo(ax, ec["Z"], ec["R_full"], ec["Ez"], ec["Er"])
        else:
            ax.streamplot(ec["Z"], ec["R_full"], ec["Ez"], ec["Er"],
                          color="0.55", density=1.4, linewidth=0.8,
                          arrowsize=1.0, arrowstyle="->")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Potential [V]")
    pc.ticks_colorbar(cbar, niveles[0], niveles[-1])
    pc.marcar_colorbar(cbar)

    ax.set_facecolor("0.93")   # fuera del dominio (membrana) = gris clarito
    if titulo:
        base = "Electric Field Lines" if con_campo else "2D Electric Potential Map"
        ax.set_title(f"{base} — {ctx['v_label']}V  |  {ctx['stem']}")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("r [nm]")

    pc.guias_canal(ax, ctx)
    pc.rectangulos_films(ax, ctx, r_max_nm=r2.max(), con_labels=True)
    if leyenda and ctx["films"]:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return fig


# =============================================================================
# CONTRATO DEL EXTRACTOR (ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None,
             titulo="2D ELECTRIC POTENTIAL MAP"):
    """→ (datos, ctx) o None. datos = {z_nm, r_nm, tri, phi_V}.
    v_label=None → pregunta por consola; con valor → valida (batch).
    sol = pc.cargar_solucion() ya cargado (batch multi-voltaje) o None.
    `titulo` permite a field_lines reusar este preparar con su banner."""
    prep = pc.preparar_comun(ruta_solucion, titulo, v_label, sol)
    if prep is None:
        return None
    sol, ctx, campos = prep
    dm    = sol["dm"]
    phi_V = campos["phi_adim"] * ctx["V_T"]
    print(f"    φ en [{phi_V.min():.4f} V, {phi_V.max():.4f} V]")
    datos = {"z_nm": dm["z_nm"], "r_nm": dm["r_nm"], "tri": dm["tri"],
             "phi_V": phi_V}
    return datos, ctx


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False,
            con_campo=False, sufijo="potential_map"):
    """Mapa limpio publication-ready (mismo output que el '¿Guardar mapa
    limpio?' de consola): {stem}_{sufijo}_{V}V.png. Sin datos tabulares
    (con_datos se acepta por uniformidad del contrato y se ignora).
    `con_campo`/`sufijo` permiten a field_lines reusar este guardar."""
    rutas = []
    if png:
        figc = crear_figura(datos, ctx, con_campo=con_campo, leyenda=False,
                            titulo=False, figsize=(10, 4))
        stem = pathlib.Path(ruta_solucion).stem
        rutas.append(pc.guardar_figura(
            figc, ruta_solucion, f"{stem}_{sufijo}_{ctx['v_label']}V.png",
            limpio=True))
    return rutas


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep

    print(">>> Generating 2D map...")
    crear_figura(datos, ctx, con_campo=False)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()

    resp = input("\nSave clean map (content + scale only, "
                 "no axes/legends)? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion)
    else:
        print("    (no image saved)")
