# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — EXTRACTOR (Pillar 3). Two usage modes.
═══════════════════════════════════════════════════════════════════════════

  python launch_extractor.py   → INTERACTIVE: opens the GUI (Streamlit)
                                        in the browser (with a fallback to the
                                        console menu if it is not installed).

  python launch_extractor.py <solution> <module> --voltage V [--save-data]
                                      → BATCH: no prompts or windows (Agg).
                                        ALWAYS saves the module's PNG(s) in the
                                        solution folder; with --save-data it
                                        also exports the numerical data
                                        (.txt/.xlsx) if the module has it.
                                        --voltage accepts:
                                            -1.0            one voltage
                                            -1.0,0.0,0.5    several (the mesh is
                                                            loaded ONLY once)
                                            all             all of the .h5
                                        (solution_summary does not use --voltage)

  python launch_extractor.py <solution>  → lists the applicable modules.
  python launch_extractor.py list        → lists the available modules.

The extractor reads a SOLUTION from the solver (Pillar 2): the <...>.h5 and its
<...>_sim.json (in the same folder). The GUI (gui_extractor_app.py) shows the
I-V curve of the full sweep and the internal physics (axial profiles, 2D maps
of potential/ions/precipitation) at each voltage.

Modules (contract in modulos/MODULE_CONTRACT.md):
  potential_map   : 2D potential + field lines
  ion_maps      : concentration of each ion (2D) + total ions
  precipitation_map : precipitation zone (only sparingly soluble salts)
  axial_profiles       : axial profiles (potential, ions, section average)
  + solution_summary/solution_summary.py (run summary)

Requires FEniCS (dolfin 2019.1.0) + matplotlib + numpy. The GUI adds
streamlit (pip install streamlit).
"""
import os
import sys

# ─── Entorno: deshabilitar MPI/UCX (los módulos son seriales) ───────────────
os.environ["OMPI_MCA_pml"] = "^ucx"
os.environ["OMPI_MCA_btl"] = "^openib"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pathlib
import json
import importlib.util
sys.dont_write_bytecode = True

AQUI            = pathlib.Path(__file__).resolve().parent
CARPETA_MODULOS = AQUI / "modulos"
CARPETA_RESUMEN = AQUI / "solution_summary"
SCRIPT_RESUMEN  = CARPETA_RESUMEN / "solution_summary.py"
APP_STREAMLIT   = AQUI / "gui_extractor_app.py"

# Historial e I/O viven en RESULTS/ (no en la carpeta del extractor).
_REPO_ROOT            = AQUI.parent
ARCHIVO_HISTORIAL     = _REPO_ROOT / "RESULTS" / ".ultima_solucion.txt"
RESULTADOS_SOLUCIONES = _REPO_ROOT / "RESULTS" / "solutions"


# =============================================================================
# DESCUBRIMIENTO DE MÓDULOS (compartido por los 2 modos)
# =============================================================================
def listar_scripts(categoria):
    """Lista los .py de una categoría (ignora los que empiezan con '__')."""
    return sorted(f for f in categoria.iterdir()
                  if f.is_file() and f.suffix == '.py' and not f.name.startswith('__'))


def todos_los_modulos():
    """Dict {nombre_modulo: ruta} de todos los módulos (todas las categorías)."""
    out = {}
    if CARPETA_MODULOS.is_dir():
        for cat in sorted(CARPETA_MODULOS.iterdir()):
            if cat.is_dir() and not cat.name.startswith(('.', '__')):
                for s in listar_scripts(cat):
                    out[s.stem] = s
    if SCRIPT_RESUMEN.exists():
        out[SCRIPT_RESUMEN.stem] = SCRIPT_RESUMEN
    return out


def _cargar_modulo(ruta_script):
    """Importa dinámicamente el .py de un módulo y devuelve el objeto módulo."""
    spec = importlib.util.spec_from_file_location(ruta_script.stem, ruta_script)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def ejecutar_modulo(ruta_script, ruta_solucion):
    """Corre procesar(ruta_solucion) del módulo, con mensajes de error claros."""
    print(f"\n>>> Launching module: {ruta_script.name} ...")
    try:
        modulo = _cargar_modulo(ruta_script)
        modulo.procesar(ruta_solucion)
    except AttributeError:
        print(f"❌ Module {ruta_script.name} does not expose procesar(ruta_solucion).")
    except Exception as e:
        print(f"❌ Something broke while running the module: {e}")


def modulo_aplica(ruta_script, meta):
    """¿El módulo aplica a esta solución? Usa aplica(meta) si el módulo la
    expone (ej. precipitación: solo sales poco solubles). Sin aplica() → True."""
    try:
        modulo = _cargar_modulo(ruta_script)
        if hasattr(modulo, "aplica"):
            return bool(modulo.aplica(meta))
    except Exception:
        return True
    return True


def cargar_meta_solucion(ruta_solucion):
    """Carga el _sim.json de la carpeta de la solución (o {} si no está)."""
    candidatos = list(ruta_solucion.parent.glob("*_sim.json"))
    if not candidatos:
        return {}
    try:
        with open(candidatos[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# MODO BATCH
# =============================================================================
def _err(msg):
    sys.exit(f"❌ {msg}")


def _resolver_solucion(ruta):
    """Acepta el .h5 o la carpeta de la solución; devuelve el Path al .h5.
    Verifica que exista el _sim.json hermano (contrato con el solver)."""
    p = pathlib.Path(os.path.expanduser(ruta))
    if not p.is_absolute():
        p = pathlib.Path(os.path.abspath(p))

    if p.is_dir():
        h5s = sorted(f for f in p.glob("Solutions_*.h5"))
        if len(h5s) != 1:
            _err(f"'{ruta}' has {len(h5s)} Solutions_*.h5 files "
                 f"(expected 1). Point to the .h5 directly.")
        p = h5s[0]
    elif not p.exists():
        _err(f"Could not find the solution '{ruta}' (cwd: {os.getcwd()}).")

    if p.suffix != ".h5":
        _err(f"'{ruta}' is not a .h5. Pass the solver's Solutions_*.h5 or its folder.")
    if not list(p.parent.glob("*_sim.json")):
        _err(f"There is no *_sim.json next to '{p.name}'. The extractor needs it "
             f"(the solver generates it). Is it a complete solution folder?")
    return p


def _imprimir_modulos():
    mods = todos_los_modulos()
    print("\nAvailable modules (use them as the 2nd argument):")
    for nombre, ruta in mods.items():
        cat = ruta.parent.name
        print(f"  {nombre:<28} ({cat})")
    print("\nExample:  python launch_extractor.py <solution.h5> axis_profile_potential"
          " --voltage -1.0 --save-data")


def _resolver_voltajes(voltaje, labels):
    """Argumento --voltaje → lista de etiquetas '+X.XX' presentes en el .h5.
    Acepta 'todos', un número o una lista '-1.0,0.0,0.5'. Los voltajes que no
    estén en el .h5 se saltean con aviso; si no queda ninguno, aborta."""
    if voltaje.strip().lower() in ("todos", "all"):
        return list(labels)
    pedidas = []
    for trozo in voltaje.split(","):
        trozo = trozo.strip()
        if not trozo:
            continue
        try:
            lab = f"{float(trozo):+.2f}"
        except ValueError:
            _err(f"--voltage: '{trozo}' is not a number (nor 'all').")
        if lab not in labels:
            print(f"[batch] NOTE: no /U_{lab}V dataset in the .h5; skipping.")
            continue
        pedidas.append(lab)
    if not pedidas:
        _err(f"None of the requested voltages are in the .h5. "
             f"Available: {[l + 'V' for l in labels]}")
    return pedidas


def _modo_batch(ruta_sol, nombre_modulo, voltaje, guardar_datos=False):
    """Corre un módulo sin prompts ni ventanas, vía su contrato
    preparar()/guardar() (ver modulos/MODULE_CONTRACT.md).
    Siempre guarda el/los PNG; con guardar_datos exporta también las tablas."""
    ruta_solucion = _resolver_solucion(ruta_sol)

    mods = todos_los_modulos()
    if nombre_modulo is None:
        print(f"[Solution]: {ruta_solucion.name}  (in {ruta_solucion.parent})")
        _imprimir_modulos()
        return
    if nombre_modulo not in mods:
        _err(f"Module '{nombre_modulo}' does not exist. Valid: {sorted(mods)}\n"
             f"   (python launch_extractor.py list)")

    # Backend sin display ANTES de cargar el módulo (los módulos importan
    # pyplot al cargarse). El batch nunca abre ventanas: solo savefig.
    import matplotlib
    matplotlib.use("Agg")

    # Filtro de aplicabilidad: en batch se SALTEA (un barrido scripteado no
    # tiene a nadie mirando; una figura sin sentido físico es peor que nada).
    meta = cargar_meta_solucion(ruta_solucion)
    if not modulo_aplica(mods[nombre_modulo], meta):
        print(f"[batch] '{nombre_modulo}' does not apply to this solution (aplica() "
              f"= False; e.g. precipitation with a fully soluble salt). Skipping.")
        return

    modulo = _cargar_modulo(mods[nombre_modulo])
    if not (hasattr(modulo, "preparar") and hasattr(modulo, "guardar")):
        _err(f"Module '{nombre_modulo}' does not implement the batch contract "
             f"(preparar/guardar; see modulos/MODULE_CONTRACT.md).\n"
             f"   Run it from the interactive menu, or add the contract to it.")

    if str(CARPETA_MODULOS) not in sys.path:
        sys.path.insert(0, str(CARPETA_MODULOS))
    import porosim_comun as pc

    rutas = []
    if not getattr(modulo, "USA_VOLTAJE", True):
        # Módulo global (solution_summary): sin voltaje ni malla.
        if voltaje is not None:
            print(f"[batch] NOTE: '{nombre_modulo}' does not use --voltage; ignored.")
        prep = modulo.preparar(ruta_solucion)
        if prep is None:
            _err(f"'{nombre_modulo}' could not prepare the solution.")
        datos, ctx = prep
        rutas = modulo.guardar(datos, ctx, ruta_solucion,
                               con_datos=guardar_datos)
    else:
        if voltaje is None:
            _err("Missing --voltage. Examples:  --voltage -1.0  |  "
                 "--voltage -1.0,0.5  |  --voltage all")
        # Carga pesada UNA sola vez (malla + espacio + voltajes disponibles);
        # el loop por voltaje solo relee los campos de cada dataset.
        print(f">>> [batch] Loading mesh from: {ruta_solucion.name}")
        sol = pc.cargar_solucion(ruta_solucion)
        if not sol["labels"]:
            _err("No /U_*V datasets in the .h5.")
        etiquetas = _resolver_voltajes(voltaje, sol["labels"])
        print(f">>> [batch] {nombre_modulo} × {len(etiquetas)} voltage(s)"
              + ("  [+ numerical data]" if guardar_datos else ""))
        for v_label in etiquetas:
            prep = modulo.preparar(ruta_solucion, v_label=v_label, sol=sol)
            if prep is None:
                print(f"[batch] {v_label}V: could not prepare; skipping.")
                continue
            datos, ctx = prep
            rutas += modulo.guardar(datos, ctx, ruta_solucion,
                                    con_datos=guardar_datos)

    print(f"\n[batch] Done: {len(rutas)} file(s) in {ruta_solucion.parent}")


# =============================================================================
# MODO CONSOLA (menús + explorador; fallback si no está Streamlit)
# =============================================================================
def cargar_solucion_gui():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    inicio = str(RESULTADOS_SOLUCIONES) if RESULTADOS_SOLUCIONES.is_dir() else str(_REPO_ROOT)
    print("\nOpening file browser...")
    ruta = filedialog.askopenfilename(
        title="Choose the solution file (.h5)",
        initialdir=inicio,
        filetypes=[("FEniCS solutions", "*.h5 *.xdmf"), ("All", "*.*")],
    )
    root.destroy()
    return pathlib.Path(ruta) if ruta else None


def obtener_ruta_solucion():
    """Ofrece la última solución usada; si no, abre el explorador (fallback a
    prompt de texto si no hay entorno gráfico)."""
    if ARCHIVO_HISTORIAL.exists():
        try:
            ruta_texto = ARCHIVO_HISTORIAL.read_text(encoding="utf-8").strip()
        except Exception:
            ruta_texto = ""
        if ruta_texto and pathlib.Path(ruta_texto).exists():
            ruta_guardada = pathlib.Path(ruta_texto)
            print(f"\n[Last solution used]: {ruta_guardada.name}")
            resp = input("Use this solution? (y/n) [Enter = y]: ").strip().lower()
            if resp in ['', 's', 'si', 'y']:
                return ruta_guardada

    try:
        ruta_nueva = cargar_solucion_gui()
    except Exception as e:
        print(f"  [NOTE] Could not open the file browser ({e}). Enter the path by hand.")
        entrada = input("Path to the solution's .h5: ").strip()
        ruta_nueva = pathlib.Path(os.path.expanduser(entrada)) if entrada else None

    if ruta_nueva:
        try:
            ARCHIVO_HISTORIAL.parent.mkdir(parents=True, exist_ok=True)
            ARCHIVO_HISTORIAL.write_text(str(ruta_nueva), encoding="utf-8")
        except Exception:
            pass
    return ruta_nueva


def seleccionar_opcion(opciones, tipo):
    if not opciones:
        print(f"\nNo {tipo}s available here.")
        return None
    print(f"\n=== Available {tipo}s ===")
    for i, item in enumerate(opciones, 1):
        print(f"[{i}] {item.name}")
    while True:
        sel = input(f"\n{tipo.capitalize()} number (or 'q' to go back): ").strip()
        if sel.lower() == 'q':
            return None
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(opciones):
                return opciones[idx]
            print("Number out of range.")
        except ValueError:
            print("Enter a valid number.")


def ofrecer_resumen(ruta_solucion):
    if not SCRIPT_RESUMEN.exists():
        return
    resp = input("\nView the summary of the open solution? (y/n) [Enter = y]: ").strip().lower()
    if resp in ['', 's', 'si', 'y']:
        ejecutar_modulo(SCRIPT_RESUMEN, ruta_solucion)
        input("\n[Enter to go to the module menu...]")


def categoria_tiene_aplicables(categoria, meta):
    return any(modulo_aplica(s, meta) for s in listar_scripts(categoria))


def _modo_consola():
    print("=== POROSIM — Extractor (Pillar 3) ===")
    if not CARPETA_MODULOS.is_dir():
        _err(f"The 'modulos' folder was not found in {AQUI}.")

    while True:   # NIVEL 1: archivo
        ruta_solucion = obtener_ruta_solucion()
        if not ruta_solucion:
            print("Operation cancelled. Exiting.")
            return
        print(f"\n[Solution loaded]: {ruta_solucion.name}")
        ofrecer_resumen(ruta_solucion)

        while True:   # NIVEL 2: categoría
            meta = cargar_meta_solucion(ruta_solucion)
            cats = [d for d in sorted(CARPETA_MODULOS.iterdir())
                    if d.is_dir() and not d.name.startswith(('.', '__'))
                    and categoria_tiene_aplicables(d, meta)]
            print(f"\n--- Menu | Solution: {ruta_solucion.name} ---")
            cat = seleccionar_opcion(cats, "category")
            if not cat:
                resp = input("\nLoad another solution? (y/n): ").strip().lower()
                if resp in ['s', 'si', 'y']:
                    break
                print("\nDone! Closing extractor.")
                return

            while True:   # NIVEL 3: módulo
                meta = cargar_meta_solucion(ruta_solucion)
                scripts = [s for s in listar_scripts(cat) if modulo_aplica(s, meta)]
                print(f"\n--- Category: {cat.name} ---")
                script = seleccionar_opcion(scripts, "module")
                if not script:
                    break
                ejecutar_modulo(script, ruta_solucion)
                input("\n[Enter to continue...]")


# =============================================================================
# MODO INTERACTIVO: GUI Streamlit (fallback → menú de consola)
# =============================================================================
def _interactivo():
    try:
        import streamlit  # noqa: F401  (solo para saber si está instalado)
        import subprocess
        print(">>> Opening the interactive extractor (Streamlit) in the browser...\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_STREAMLIT)],
                       cwd=str(AQUI))
    except ImportError:
        print("\n>>> Streamlit is not installed in this Python environment.")
        print("    (You can install it with: pip install streamlit for the graphical interface)\n")
        _modo_consola()


# =============================================================================
if __name__ == "__main__":
    argv = sys.argv[1:]

    voltaje = None
    if "--voltage" in argv:
        i = argv.index("--voltage")
        if i + 1 >= len(argv):
            _err("--voltage needs a value:  --voltage -1.0 | -1.0,0.5 | all")
        voltaje = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    guardar_datos = "--save-data" in argv
    if guardar_datos:
        argv.remove("--save-data")

    if argv and argv[0] in ("list", "--list"):
        _imprimir_modulos()
    elif argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__)
    elif argv and (argv[0].endswith(".h5") or pathlib.Path(argv[0]).is_dir()):
        nombre_modulo = argv[1] if len(argv) >= 2 else None
        _modo_batch(argv[0], nombre_modulo, voltaje, guardar_datos)
    elif argv:
        _err(f"Unrecognized argument: '{argv[0]}'.\n"
             f"   python launch_extractor.py [solution.h5 [module] "
             f"[--voltage V] [--save-data]]\n"
             f"   python launch_extractor.py list        (available modules)")
    else:
        _interactivo()
