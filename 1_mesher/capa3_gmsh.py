# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MALLADOR · CAPA 3: EMISIÓN A GMSH (geometría real + grupos físicos)
  Traduce el modelo (Capa 1) + los curve loops orientados (Capa 2) a
  geometría REAL de Gmsh. Por ahora SOLO CONSTRUYE Y VERIFICA: no guarda
  archivos ni abre la GUI.
═══════════════════════════════════════════════════════════════════════════

  Requiere en la MISMA carpeta:
      capa1_modelo.py   (modelo: puntos, líneas, superficies)
      capa2_loops.py   (loops orientados + verificación)

  QUÉ HACE
  ────────
  Para una config (ticks de film), construye en Gmsh:
    1. los PUNTOS (addPoint), guardando nombre_lógico → tag_gmsh
    2. las LÍNEAS rectas (addLine) y la pared del cono (addSpline)
    3. los CURVE LOOPS (addCurveLoop) con los signos de la Capa 2
    4. las SUPERFICIES (addPlaneSurface)
    5. los GRUPOS FÍSICOS (addPhysicalGroup), agrupando por tag:
         - AXIS junta TODAS las líneas de eje (una por slab)
         - DOMAIN_FLUID junta TODAS las superficies de fluido
         - cada tag de línea/film va a su grupo

  Luego sincroniza y comprueba que Gmsh no reporte errores de topología
  (el equivalente al "Curve loop N is wrong"). NO mallaa, NO guarda, NO GUI.

  Validación: corre los 4 casos y reporta OK/FALLA de cada uno.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gmsh

from capa1_modelo import (
    Params, construir_estaciones, construir_slabs, construir_puntos,
    construir_lineas, construir_superficies, R_canal, TAGS,
)
from capa2_loops import (
    construir_loops, indexar_lineas, verificar_cierre, verificar_bordes_compartidos,
)

# Tamaño nominal de punto: irrelevante para la geometría (el refinamiento es
# cosa del mesher, capa futura). addPoint lo pide, nada más.
LC_PLACEHOLDER = 1.0e-9


def emitir_geometria(p: Params):
    """Construye la geometría completa en Gmsh para la config dada.
    Devuelve un dict con info para diagnóstico. NO sincroniza fuera de lo
    necesario, NO mallea, NO guarda."""

    # --- modelo (Capas 1 y 2) ---
    estaciones = construir_estaciones(p)
    slabs      = construir_slabs(estaciones, p)
    puntos     = construir_puntos(estaciones, p)        # nombre → Punto(z, r)
    lineas     = construir_lineas(estaciones, slabs, p) # lista de Linea
    superficies = construir_superficies(slabs)          # [(tag_name, corrida)]
    idx_lineas = indexar_lineas(lineas)
    loops      = construir_loops(superficies)           # [(tag_name, corrida, loop)]

    # --- arrancar Gmsh limpio ---
    if gmsh.isInitialized():
        gmsh.finalize()
    # interruptible=False evita que Gmsh instale un manejador de signal.SIGINT,
    # que falla fuera del hilo principal (p.ej. corriendo dentro de Streamlit:
    # "signal only works in main thread"). Si la versión de Gmsh no acepta el
    # argumento, se cae a initialize() normal.
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    gmsh.model.add("dibujo")
    gmsh.option.setNumber("General.Terminal", 0)        # silenciar consola Gmsh
    gmsh.option.setNumber("Geometry.Tolerance",        1e-11)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-11)

    # ============================================================
    # 1) PUNTOS  (nombre_lógico → tag de Gmsh)
    # ============================================================
    gtag_punto = {}
    for nombre, pt in puntos.items():
        gtag_punto[nombre] = gmsh.model.geo.addPoint(pt.z, pt.r, 0.0, LC_PLACEHOLDER)

    # ============================================================
    # 2) LÍNEAS  (rectas con addLine; la pared con addSpline)
    #    nombre_lógico → tag de Gmsh
    # ============================================================
    # Puntos intermedios de la pared del cono (no son puntos de estación):
    # se crean acá para la spline. Los extremos SÍ son p_tip_mouth / p_base_mouth.
    # Para bullet: muestreo adaptativo (muchos puntos cerca del tip, pocos hacia la base).
    # Para lineal: muestreo uniforme (cónico/cilindro no lo necesita).
    spline_pts = [gtag_punto["p_tip_mouth"]]
    if getattr(p, "channel_type", "conical") == "bullet":
        # Distribución potencial: concentra en x pequeño (tip), dilata hacia la base.
        # y = u^2 comprime el inicio; u va de 0 a 1.
        u = np.linspace(0.0, 1.0, p.N_PTS_WALL)
        xs = u * u * p.L_pore  # cuadrática: puntos densos al inicio
    else:
        xs = np.linspace(0.0, p.L_pore, p.N_PTS_WALL)
    for x in xs[1:-1]:
        z = p.z_tip + x
        r = R_canal(p, z)
        spline_pts.append(gmsh.model.geo.addPoint(z, r, 0.0, LC_PLACEHOLDER))
    spline_pts.append(gtag_punto["p_base_mouth"])

    gtag_linea = {}
    for l in lineas:
        if l.nombre == "l_wall":
            gtag_linea[l.nombre] = gmsh.model.geo.addSpline(spline_pts)
        else:
            gtag_linea[l.nombre] = gmsh.model.geo.addLine(
                gtag_punto[l.p0], gtag_punto[l.p1])

    # ============================================================
    # 3 y 4) CURVE LOOPS + SUPERFICIES
    #    El signo de la Capa 2 se traduce a +tag / -tag de Gmsh.
    # ============================================================
    sup_por_tag = {}   # tag_name → [tags de superficie de Gmsh]
    for tag_name, corrida, loop in loops:
        curvas_firmadas = [signo * gtag_linea[nombre] for signo, nombre in loop]
        cl = gmsh.model.geo.addCurveLoop(curvas_firmadas)
        s  = gmsh.model.geo.addPlaneSurface([cl])
        sup_por_tag.setdefault(tag_name, []).append(s)

    gmsh.model.geo.synchronize()

    # ============================================================
    # 5) GRUPOS FÍSICOS (agrupados por tag)
    # ============================================================
    # --- líneas: juntar por tag físico ---
    lineas_por_tag = {}   # tag_int → (nombre_tag, [tags de línea de Gmsh])
    # Mapa inverso tag_int → nombre, para nombrar el grupo
    nombre_de_tag = {v: k for k, v in TAGS.items()}
    for l in lineas:
        if l.tag is None:
            continue
        lineas_por_tag.setdefault(l.tag, []).append(gtag_linea[l.nombre])
    for tag_int, glineas in lineas_por_tag.items():
        gmsh.model.addPhysicalGroup(1, glineas, tag_int,
                                    name=nombre_de_tag[tag_int])

    # --- superficies: juntar por tag (DOMAIN_FLUID puede ser varias) ---
    for tag_name, gsups in sup_por_tag.items():
        gmsh.model.addPhysicalGroup(2, gsups, TAGS[tag_name], name=tag_name)

    return {
        "n_puntos": len(gtag_punto) + max(0, p.N_PTS_WALL - 2),
        "n_lineas": len(gtag_linea),
        "n_superficies": sum(len(v) for v in sup_por_tag.values()),
        "tags_superficie": {k: len(v) for k, v in sup_por_tag.items()},
        "loops": loops,
        "idx_lineas": idx_lineas,
    }


def validar_config(p: Params, titulo=""):
    """Verifica en Python (cierre + bordes) y emite a Gmsh, capturando errores."""
    print("\n" + "=" * 70)
    print(f"  CONFIG: {titulo}")
    print("=" * 70)

    # --- 1) verificación en Python (rápida, antes de tocar Gmsh) ---
    estaciones = construir_estaciones(p)
    slabs      = construir_slabs(estaciones, p)
    lineas     = construir_lineas(estaciones, slabs, p)
    superficies = construir_superficies(slabs)
    idx        = indexar_lineas(lineas)
    loops      = construir_loops(superficies)

    py_ok = True
    for tag_name, _, loop in loops:
        ok, msg, _ = verificar_cierre(loop, idx)
        if not ok:
            print(f"  [Python] ✗ loop {tag_name}: {msg}")
            py_ok = False
    ok_bordes, problemas, _ = verificar_bordes_compartidos(loops)
    if not ok_bordes:
        py_ok = False
        for pr in problemas:
            print(f"  [Python] ✗ borde: {pr}")
    print(f"  [Python] verificación de loops: {'✓ OK' if py_ok else '✗ FALLA'}")

    # --- 2) emisión real a Gmsh ---
    try:
        info = emitir_geometria(p)
        # Si llegó acá sin excepción, Gmsh construyó y sincronizó bien.
        print(f"  [Gmsh]   construcción: ✓ OK")
        print(f"           puntos={info['n_puntos']}  líneas={info['n_lineas']}  "
              f"superficies={info['n_superficies']}")
        print(f"           superficies por tag: {info['tags_superficie']}")
        gmsh_ok = True
    except Exception as e:
        print(f"  [Gmsh]   construcción: ✗ FALLA → {e}")
        gmsh_ok = False
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

    return py_ok and gmsh_ok


def inspeccionar_gui(p: Params, titulo=""):
    """Reconstruye la config y abre la GUI de Gmsh para inspección visual.
    Muestra la geometría (puntos, líneas, superficies). Si se descomenta el
    bloque de mallado, además triangula con elementos GRANDES (rápido) para
    ver los colores de los dominios. Cerrá la ventana para volver."""
    print(f"\n>>> Abriendo Gmsh para inspeccionar: {titulo}")
    print("    (cerrá la ventana de Gmsh para volver a la consola)")

    info = emitir_geometria(p)   # construye la geometría y deja Gmsh inicializado

    # Activar terminal de Gmsh para ver mensajes en la GUI
    gmsh.option.setNumber("General.Terminal", 1)

    # ----------------------------------------------------------------
    # (OPCIONAL) MALLA 2D GRUESA — descomentar para ver los dominios
    # coloreados. Elementos grandes a propósito: es solo para inspección
    # visual rápida, NO es la malla final (esa lleva refinamiento, capa futura).
    # Para quitarla, volvé a comentar este bloque.
    # ----------------------------------------------------------------
    # gmsh.option.setNumber("Mesh.MeshSizeMin", 20e-9)   # elemento mínimo ~20 nm
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 60e-9)   # elemento máximo ~60 nm
    # gmsh.option.setNumber("Mesh.Algorithm", 6)         # Frontal-Delaunay
    # gmsh.option.setNumber("Mesh.SurfaceFaces", 1)      # mostrar caras de malla
    # gmsh.option.setNumber("Mesh.ColorCarousel", 2)     # colorear por dominio físico
    # gmsh.model.mesh.generate(2)
    # ----------------------------------------------------------------

    try:
        gmsh.fltk.run()    # abre la ventana; bloquea hasta que la cerrás
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


if __name__ == "__main__":
    casos = [
        ("sin film",  Params(include_film_tip=False, include_film_base=False)),
        ("film tip",  Params(include_film_tip=True,  include_film_base=False)),
        ("film base", Params(include_film_tip=False, include_film_base=True)),
        ("dos films", Params(include_film_tip=True,  include_film_base=True)),
    ]
    resultados = []
    for titulo, p in casos:
        resultados.append((titulo, validar_config(p, titulo)))

    print("\n" + "=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    for titulo, ok in resultados:
        print(f"    {titulo:<12} {'✓ OK' if ok else '✗ FALLA'}")
    print()

    # --- Inspección visual opcional en la GUI de Gmsh ---
    print("=" * 70)
    print("  ¿Inspeccionar algún caso en la GUI de Gmsh?")
    for i, (titulo, _) in enumerate(casos, 1):
        print(f"    [{i}] {titulo}")
    print("    [Enter] no, terminar")
    sel = input("  Opción: ").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(casos):
        titulo, p = casos[int(sel) - 1]
        inspeccionar_gui(p, titulo)
    else:
        print("  Listo, sin inspección.")