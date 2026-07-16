# -*- coding: utf-8 -*-
"""
Módulo: Líneas de Campo Eléctrico
Categoría: potential_map

Genera líneas de campo E = -grad(phi) (con flechas, para ver dirección del
movimiento de iones) sobre un fondo de mapa de potencial 2D.

El campo se obtiene con matplotlib.tri.CubicTriInterpolator directamente sobre
la malla (sin project()): da gradientes suaves y enmascara automáticamente
fuera del dominio (respeta paredes y cono). No usa muestreo punto-por-punto.

Este módulo es el mapa de potencial CON líneas de campo: comparte toda la
lógica con potential.py (crear_figura(..., con_campo=True) y el contrato
preparar / guardar). Editar el mapa base en potential.py repercute acá
también.

Funciona con o sin films. Cada film se dibuja como un borde:
    rojo  si la carga fija es positiva
    azul  si la carga fija es negativa

Convenciones: ver README.md (Pilar 3) y MODULE_CONTRACT.md
"""
import sys
import pathlib

import matplotlib.pyplot as plt

_AQUI = pathlib.Path(__file__).resolve().parent
_MOD  = str(_AQUI.parent)
if _MOD not in sys.path:
    sys.path.insert(0, _MOD)
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))
import porosim_comun as pc          # noqa: F401  (coherencia; lo usa potencial)
import potential                    # mismo directorio (potential_map/)

# Re-exportar la API pura: crear_figura con líneas de campo por defecto y el
# cálculo del campo, para que la GUI pueda importar cualquiera de los dos.
campo_electrico = potential.campo_electrico


def crear_figura(datos, ctx, con_campo=True, **kwargs):
    """Mapa de potencial + líneas de campo (con_campo=True por defecto)."""
    return potential.crear_figura(datos, ctx, con_campo=con_campo, **kwargs)


# =============================================================================
# CONTRATO DEL EXTRACTOR (delega en potential.py: mismo mapa, con campo)
# =============================================================================
def preparar(ruta_solucion, v_label=None, sol=None):
    return potential.preparar(ruta_solucion, v_label, sol,
                              titulo="ELECTRIC FIELD LINES")


def guardar(datos, ctx, ruta_solucion, png=True, con_datos=False):
    return potential.guardar(datos, ctx, ruta_solucion, png=png,
                             con_datos=con_datos, con_campo=True,
                             sufijo="field_lines")


# =============================================================================
# CÁSCARA DE CONSOLA (menú interactivo)
# =============================================================================
def procesar(ruta_solucion):
    prep = preparar(ruta_solucion)
    if prep is None:
        return
    datos, ctx = prep

    print(">>> Computing electric field E = -grad(φ) and generating plot...")
    crear_figura(datos, ctx, con_campo=True)
    print(">>> Opening window. Close it to return to the menu.")
    plt.show()

    resp = input("\nSave clean map (content + scale only, "
                 "no axes/legends)? [y/N]: ").strip().lower()
    if resp in ("s", "si", "sí", "y", "yes"):
        guardar(datos, ctx, ruta_solucion)
    else:
        print("    (no image saved)")
