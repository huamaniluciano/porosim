# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM WEB — NATIVE MULTIPAGE HUB AND CONTAINER
  Container skeleton that runs the pillars' GUIs in real time.

  Launch:
      python launch_porosim.py
      # or:
      streamlit run launch_porosim.py

  Without Streamlit installed, it falls back to a console menu that routes to
  the terminal mode of each pillar (interactive solver, extractor menus, mesher
  batch help).
═══════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import pathlib

# ─── Entorno limpio ANTES de importar FEniCS o librerías científicas ──────
os.environ["OMPI_MCA_pml"] = "^ucx"
os.environ["OMPI_MCA_btl"] = "^openib"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

try:
    import h5py
except Exception:
    pass

# ─── Configurar sys.path para que cada pilar encuentre sus módulos hermanos ──
RAIZ_ZENODO = pathlib.Path(__file__).resolve().parent
for sub in ("1_mesher", "2_solver", "3_extractor"):
    p = str(RAIZ_ZENODO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
if str(RAIZ_ZENODO) not in sys.path:
    sys.path.insert(0, str(RAIZ_ZENODO))


def esta_en_streamlit():
    """Detecta si el script se está ejecutando dentro del servidor Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _fallback_consola():
    """Sin Streamlit: menú de terminal que deriva al modo consola de cada
    pilar. Cada opción corre el script correspondiente en un subproceso."""
    import subprocess
    print("Streamlit is not installed (pip install streamlit for the web UI).")
    print("Available console modes:\n")
    opciones = {
        "1": ("Mesher    — batch mode guide (geom.json)",
              [str(RAIZ_ZENODO / "1_mesher" / "launch_mallador.py"), "help"]),
        "2": ("Solver    — interactive console prompts",
              [str(RAIZ_ZENODO / "2_solver" / "solver.py")]),
        "3": ("Extractor — file browser + console menus",
              [str(RAIZ_ZENODO / "3_extractor" / "console_backup" / "extractor.py")]),
    }
    while True:
        for k, (desc, _) in opciones.items():
            print(f"  [{k}] {desc}")
        sel = input("\nChoose a pillar (or 'q' to quit): ").strip().lower()
        if sel in ("q", ""):
            print("Exiting.")
            return
        if sel in opciones:
            subprocess.run([sys.executable] + opciones[sel][1])
            print()
        else:
            print("Invalid option.\n")


if not esta_en_streamlit():
    import subprocess
    print("\n" + "═" * 70)
    print("   POROSIM WEB · Integrated PNP Scientific Suite (Native Streamlit)")
    print("═" * 70)
    try:
        import streamlit  # noqa: F401  (solo para saber si está instalado)
    except ImportError:
        _fallback_consola()
        sys.exit(0)
    print(">>> Launching the Streamlit web interface in the browser...\n")
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Interrupted] Returning to the console.")
    print("\n<<< PoroSim Web server stopped.\n")
    sys.exit(0)


# =============================================================================
# CONTENEDOR NATIVO DE STREAMLIT (ESQUELETO VACÍO)
# =============================================================================
import streamlit as st

p_inicio = st.Page("portada_porosim.py", title="🏠 Home · PoroSim Hub", url_path="home", default=True)
p1       = st.Page("1_mesher/gui_app.py", title="1. Mesher", icon="📐", url_path="mesher")
p2       = st.Page("2_solver/gui_app.py", title="2. PNP Solver", icon="⚡", url_path="solver")
p3       = st.Page("3_extractor/gui_extractor_app.py", title="3. Extractor", icon="📊", url_path="extractor")

# Menú de navegación nativo de Streamlit
pg = st.navigation({
    "Welcome": [p_inicio],
    "PoroSim Pillars": [p1, p2, p3]
})

# Ejecuta en vivo la página/GUI seleccionada por el usuario
pg.run()
