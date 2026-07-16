# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — PNP SOLVER · launcher with GUI (variant of solver.py)
═══════════════════════════════════════════════════════════════════════════

  python launch_solver.py             → INTERACTIVE: opens the web GUI
                                               (Streamlit) in the browser.
                                               Without streamlit installed, it
                                               falls back to the console prompts.

  python launch_solver.py params.json [--mesh PATH]
  python launch_solver.py help
      → identical to solver.py (they delegate to it; no logic is duplicated here).

Only difference from solver.py: the no-argument mode opens the GUI instead of
the console prompts. All batch validation, help and the engine live in
solver.py / motor_pnp.py — this file is just a launcher.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "console_backup"))   # preguntas.py

# Reusar TODO de solver.py: ayuda y batch (validación incluida). Si algo se
# corrige allá, este lanzador lo hereda automáticamente (cero duplicación).
from solver import _generar_batch, _mostrar_ayuda


def _interactivo_gui():
    """GUI web (Streamlit); si no está instalado, preguntas de consola."""
    try:
        import streamlit  # noqa: F401  (solo para detectar si está instalado)
        import subprocess
        app = os.path.join(AQUI, "gui_app.py")
        print(">>> Opening the interactive solver (Streamlit) in the browser...\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app], cwd=AQUI)
    except ImportError:
        print("\n>>> Streamlit is not installed (pip install streamlit for the GUI).")
        print("    Continuing with the console prompts.\n")
        print("=" * 60)
        print("   POROSIM — PNP SOLVER (console)")
        print("=" * 60)
        from preguntas import armar_config
        cfg = armar_config()
        from motor_pnp import resolver
        resolver(cfg)


if __name__ == "__main__":
    argv = sys.argv[1:]

    malla_override = None
    if "--mesh" in argv:
        i = argv.index("--mesh")
        if i + 1 >= len(argv):
            sys.exit("❌ --mesh needs a path:  python launch_solver.py params.json --mesh PATH")
        malla_override = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    arg = argv[0] if argv else None
    if arg and arg.endswith(".json"):
        _generar_batch(arg, malla_override)
    elif malla_override:
        sys.exit("❌ --mesh only goes together with a params.json")
    elif arg in ("-h", "--help", "help", "fields", "--fields"):
        _mostrar_ayuda()
    elif arg:
        sys.exit(f"❌ Unrecognized argument: '{arg}'.\n"
                 f"   Usage:  python launch_solver.py [params.json] [--mesh PATH]")
    else:
        _interactivo_gui()
