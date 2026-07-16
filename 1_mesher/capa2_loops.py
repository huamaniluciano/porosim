# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MALLADOR · CAPA 2: LOOPS (curve loops orientados + verificación)
  Plantillas rígidas (opción B). SIN Gmsh todavía: arma los loops con signos
  y VERIFICA su cierre + la consistencia de bordes compartidos en Python.
═══════════════════════════════════════════════════════════════════════════

  Requiere capa1_modelo.py en la MISMA carpeta (provee el modelo:
  estaciones, slabs, puntos, líneas, superficies).

  QUÉ HACE ESTA CAPA
  ──────────────────
  Para cada superficie (corrida de slabs fusionada), ensambla su curve loop:
  una lista ordenada de (signo, nombre_de_línea). El signo es + si la línea se
  recorre en su dirección nativa (la de Capa 1) y − si al revés.

  Tres plantillas rígidas, según el tipo de superficie:
    • RESERVORIO PURO  (un slab fluido sin canal)
    • FILM             (un slab de film)
    • FLUIDO CON CANAL (corrida fluida que incluye el slab canal; puede tener
                        reservorios a uno o ambos lados, fusionados)

  Recorrido CW de toda plantilla: cara izquierda (sube) → techo/pared (izq→der)
  → cara derecha (baja) → eje (der→izq).

  VERIFICACIONES (lo que reemplaza al "Curve loop N is wrong" de Gmsh):
    1. CIERRE: la salida de cada línea = entrada de la siguiente; la última
       cierra sobre la primera.
    2. BORDES COMPARTIDOS: cada arista interior (interfaz film/fluido, apertura
       junto a un film) aparece en EXACTAMENTE 2 loops con signos opuestos; el
       resto, en 1 solo loop. Esto garantiza orientaciones de normal coherentes.

  El caso "film tip" reproduce 1:1 los loops del dibujador validado
  (cl_res_tip, cl_film_tip, cl_res_base), incluidos los signos negativos
  -l_film_tip_face y -l_tip_aperture.

  Próximo paso (Capa 3): emitir a Gmsh (addCurveLoop/addPlaneSurface) y tags.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capa1_modelo import (
    Params, construir_estaciones, construir_slabs,
    construir_lineas, construir_superficies, slab_suffix,
)


# =================================================================
# HELPERS DE CARA
# =================================================================
# Caras "simples" (una sola línea de axis a top) y su dirección nativa.
_NATIVO_SIMPLE = {
    "INLET":     ("l_inlet",          "sube"),   # nativo axis→top
    "OUTLET":    ("l_outlet",         "baja"),   # nativo top→axis
    "FILM_TIP":  ("l_film_tip_face",  "baja"),   # nativo top→axis
    "FILM_BASE": ("l_film_base_face", "baja"),   # nativo top→axis
}


def cara_simple(station_name, sentido):
    """Línea (con signo) de una cara simple recorrida 'sube' (axis→top) o
    'baja' (top→axis)."""
    nombre, nativo = _NATIVO_SIMPLE[station_name]
    signo = +1 if nativo == sentido else -1
    return [(signo, nombre)]


def cara_membrana(station_name, sentido, incluir_apertura):
    """Bandas (con signo) de una cara de membrana (TIP/BASE).
    'sube' recorre de adentro (aperture/charge) hacia afuera (outer);
    'baja' al revés. La apertura se incluye solo si la cara bordea un film."""
    sk = station_name.lower()
    nativo = "baja" if station_name == "TIP" else "sube"   # TIP baja, BASE sube
    signo = +1 if nativo == sentido else -1
    bandas = (["aperture"] if incluir_apertura else []) + ["charge", "far", "outer"]
    if sentido == "baja":
        bandas = list(reversed(bandas))
    return [(signo, f"l_{sk}_{b}") for b in bandas]


# =================================================================
# PLANTILLAS DE LOOP (rígidas, por tipo de región)
# =================================================================
def loop_reservorio_puro(corrida):
    """Un solo slab reservorio fluido (sin canal). Loop simple de 4 lados."""
    slab = corrida[0]
    suf = slab_suffix(slab)
    loop  = cara_simple(slab.izq.name, "sube")
    loop += [(+1, f"l_top_{suf}")]
    loop += cara_simple(slab.der.name, "baja")
    loop += [(+1, f"l_axis_{suf}")]
    return loop


def loop_film(corrida):
    """Un slab de film. La cara de MEMBRANA está del lado del canal:
       film tip  → membrana a la DERECHA (cara TIP);
       film base → membrana a la IZQUIERDA (cara BASE)."""
    slab = corrida[0]
    suf = slab_suffix(slab)
    if slab.name == "FILM_TIP":          # (FILM_TIP → TIP)
        loop  = cara_simple("FILM_TIP", "sube")           # -l_film_tip_face
        loop += [(+1, f"l_top_{suf}")]                    # l_top_film_tip
        loop += cara_membrana("TIP", "baja", incluir_apertura=True)
        loop += [(+1, f"l_axis_{suf}")]                   # l_axis_film_tip
    else:                                # FILM_BASE: (BASE → FILM_BASE)
        loop  = cara_membrana("BASE", "sube", incluir_apertura=True)
        loop += [(+1, f"l_top_{suf}")]                    # l_top_film_base
        loop += cara_simple("FILM_BASE", "baja")          # +l_film_base_face
        loop += [(+1, f"l_axis_{suf}")]                   # l_axis_film_base
    return loop


def loop_fluido_canal(corrida):
    """Corrida fluida que incluye el slab CANAL (con reservorios opcionales a
    los lados). Recorre: lado izq (sube) → pared del cono → lado der (baja) →
    eje (der→izq, un tramo por slab)."""
    primer, ultimo = corrida[0], corrida[-1]
    empieza_res = (primer.name == "RESERVORIO")   # L == INLET
    termina_res = (ultimo.name == "RESERVORIO")   # R == OUTLET

    loop = []

    # --- LADO IZQUIERDO (sube hasta la boca tip) ---
    if empieza_res:
        loop += [(+1, "l_inlet"), (+1, f"l_top_{slab_suffix(primer)}")]
        loop += cara_membrana("TIP", "baja", incluir_apertura=False)   # top→mouth
    else:
        loop += [(-1, "l_tip_aperture")]      # film a la izquierda: sube apertura

    # --- PARED DEL CONO ---
    loop += [(+1, "l_wall")]                  # tip_mouth → base_mouth

    # --- LADO DERECHO (baja hasta el eje) ---
    if termina_res:
        loop += cara_membrana("BASE", "sube", incluir_apertura=False)  # mouth→top
        loop += [(+1, f"l_top_{slab_suffix(ultimo)}"), (+1, "l_outlet")]
    else:
        loop += [(-1, "l_base_aperture")]     # film a la derecha: baja apertura

    # --- EJE (de derecha a izquierda, un tramo por slab) ---
    for s in reversed(corrida):
        loop += [(+1, f"l_axis_{slab_suffix(s)}")]

    return loop


def construir_loops(superficies):
    """Devuelve [(tag_name, corrida, loop)] aplicando la plantilla que toca."""
    out = []
    for tag_name, corrida in superficies:
        if tag_name.startswith("DOMAIN_FILM"):
            loop = loop_film(corrida)
        elif any(s.name == "CANAL" for s in corrida):
            loop = loop_fluido_canal(corrida)
        else:
            loop = loop_reservorio_puro(corrida)
        out.append((tag_name, corrida, loop))
    return out


# =================================================================
# VERIFICACIONES
# =================================================================
def indexar_lineas(lineas):
    return {l.nombre: l for l in lineas}


def verificar_cierre(loop, idx):
    """Sigue los puntos del loop. Devuelve (ok, msg, secuencia_puntos)."""
    seq = []
    for k, (signo, nombre) in enumerate(loop):
        if nombre not in idx:
            return False, f"línea inexistente: {nombre}", seq
        l = idx[nombre]
        a, b = (l.p0, l.p1) if signo > 0 else (l.p1, l.p0)
        if k == 0:
            seq.append(a)
        elif a != seq[-1]:
            return False, f"discontinuidad antes de {nombre}: tras {seq[-1]} viene {a}", seq
        seq.append(b)
    if seq[-1] != seq[0]:
        return False, f"no cierra: termina en {seq[-1]}, empezó en {seq[0]}", seq
    return True, "OK", seq


def verificar_bordes_compartidos(loops):
    """Cada arista que aparece en >1 loop debe aparecer EXACTAMENTE 2 veces con
    signos opuestos. Devuelve (ok, lista_de_problemas, dict_apariciones)."""
    apariciones = {}   # nombre → lista de signos
    for _, _, loop in loops:
        for signo, nombre in loop:
            apariciones.setdefault(nombre, []).append(signo)

    problemas = []
    for nombre, signos in apariciones.items():
        if len(signos) == 1:
            continue
        if len(signos) != 2:
            problemas.append(f"{nombre}: aparece {len(signos)} veces (debería 1 o 2)")
        elif signos[0] * signos[1] >= 0:
            problemas.append(f"{nombre}: aparece 2 veces con MISMO signo {signos}")
    return (len(problemas) == 0), problemas, apariciones


# =================================================================
# DUMP
# =================================================================
def _loop_str(loop):
    return "  ".join(f"{'+' if s > 0 else '−'}{n}" for s, n in loop)


def analizar(p: Params, titulo=""):
    est   = construir_estaciones(p)
    slabs = construir_slabs(est, p)
    lineas = construir_lineas(est, slabs, p)
    sups  = construir_superficies(slabs)
    idx   = indexar_lineas(lineas)
    loops = construir_loops(sups)

    print("\n" + "=" * 78)
    print(f"  CONFIG: {titulo}   (film_tip={p.include_film_tip}, film_base={p.include_film_base})")
    print("=" * 78)

    todo_ok = True

    print(f"\n  LOOPS ({len(loops)}):")
    for tag_name, corrida, loop in loops:
        tramos = "+".join(f"{s.izq.name}→{s.der.name}" for s in corrida)
        ok, msg, _ = verificar_cierre(loop, idx)
        estado = "✓ cierra" if ok else f"✗ {msg}"
        todo_ok = todo_ok and ok
        print(f"\n    [{tag_name}]  ({tramos})   {estado}")
        print(f"      {_loop_str(loop)}")

    ok_bordes, problemas, apar = verificar_bordes_compartidos(loops)
    todo_ok = todo_ok and ok_bordes
    print("\n  BORDES COMPARTIDOS (deben aparecer 2× con signos opuestos):")
    compartidas = {n: s for n, s in apar.items() if len(s) > 1}
    if not compartidas:
        print("    (ninguna — config sin interfaces internas)")
    for n, s in sorted(compartidas.items()):
        signos = ", ".join("+" if x > 0 else "−" for x in s)
        print(f"    {n:<20} [{signos}]")
    if problemas:
        print("    PROBLEMAS:")
        for pr in problemas:
            print(f"      ✗ {pr}")

    print(f"\n  RESULTADO: {'✓ TODO OK' if todo_ok else '✗ HAY ERRORES'}")
    return todo_ok


if __name__ == "__main__":
    casos = [
        ("sin film",  Params(include_film_tip=False, include_film_base=False)),
        ("film tip",  Params(include_film_tip=True,  include_film_base=False)),
        ("film base", Params(include_film_tip=False, include_film_base=True)),
        ("dos films", Params(include_film_tip=True,  include_film_base=True)),
    ]
    resultados = []
    for titulo, p in casos:
        resultados.append((titulo, analizar(p, titulo)))

    print("\n" + "=" * 78)
    print("  RESUMEN FINAL")
    print("=" * 78)
    for titulo, ok in resultados:
        print(f"    {titulo:<12} {'✓ OK' if ok else '✗ FALLA'}")