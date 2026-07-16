# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — SOLVER · PREGUNTAS (modo interactivo)
  Secciones 4-5 del solver original: los prompts, con los MISMOS textos.
  armar_config() devuelve el mismo dict cfg que produce el modo batch
  (params.json), así el motor no distingue de dónde vinieron los valores.
═══════════════════════════════════════════════════════════════════════════

  Sin FEniCS: este módulo solo junta valores. La física vive en motor_pnp.

  TODO(gui-solver): a futuro este módulo puede reemplazarse por una GUI web
  (como la del mallador) que devuelva el MISMO dict cfg que armar_config().
  Idea: mostrar la geometría del canal (reservorios + films) mientras se
  configura — dónde va la carga y su signo, magnitud codificada en la
  intensidad del color del film/pared — y un "proto-plot" de la I-V esperada
  (solo el dominio ±V_max con los n_steps puntos marcados, sin resolver).
  El motor no se toca: la GUI es solo otro productor de cfg.
"""
import os
import json
import sys

# Este módulo vive en 2_solver/console_backup/; constantes.py está en la raíz
# del pilar (el padre), así que la agregamos al path antes de importar.
_PILAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PILAR not in sys.path:
    sys.path.insert(0, _PILAR)

from constantes import (
    E_CHARGE, E_NM3_TO_MOLAR, film_tipos,
    C0_DEFAULT_MM, V_MAX_DEFAULT_V, N_STEPS_DEFAULT, T_DEFAULT_K,
    EPS_R_DEFAULT, SIGMA_DEFAULT_E_NM2, SIGMA_DEFAULT_CM2, FILM_TIPO_DEFAULT,
    SAL_DEFAULT, RESULTADOS_MALLAS, RESULTADOS_SOLUCIONES, HISTORIAL_MALLA,
    cargar_catalogo_sales, derivar_malla,
)


# =============================================================================
# HELPERS DE PROMPT (verbatim del solver original)
# =============================================================================
def pedir_valor(texto, valor_default, tipo=float):
    """Pide un valor por consola; si el usuario aprieta Enter, usa el default."""
    respuesta = input(f"{texto} [Enter = {valor_default}]: ").strip()
    if not respuesta:
        return valor_default
    return tipo(respuesta)


def preguntar_ok(texto):
    """Devuelve True si el usuario acepta (Enter o s/si/y), False si responde n."""
    while True:
        r = input(f"{texto} [Enter=Yes / n=change]: ").strip().lower()
        if r in ['', 's', 'si', 'y', 'yes']:
            return True
        elif r in ['n', 'no']:
            return False
        else:
            print("    [Please answer 'y' (for yes/enter) or 'n' (for no/change)]")


def elegir_tipo_film(lado, default_tipo=FILM_TIPO_DEFAULT):
    """
    Pregunta interactivamente el tipo de carga de UN film. Acepta una clave
    del catálogo ("4M") o un número en MOLAR. Devuelve (tipo, n_e_per_nm3).
    """
    print(f"    Available types: {', '.join(film_tipos.keys())}")
    print(f"    Or a number in MOLAR (e.g. 3 → 3M of fixed charge)")
    entrada = input(f"  Film '{lado}': type? [Enter={default_tipo}]: ").strip()

    if not entrada:
        return default_tipo, film_tipos[default_tipo]
    elif entrada in film_tipos:
        return entrada, film_tipos[entrada]
    else:
        try:
            c_fix_molar = float(entrada)                     # usuario ingresa en MOLAR
            return f"custom_{c_fix_molar:.2f}M", c_fix_molar / E_NM3_TO_MOLAR
        except ValueError:
            print(f"  [WARNING] Unrecognized input, using {default_tipo}")
            return default_tipo, film_tipos[default_tipo]


def elegir_sal(catalogo, default_nombre):
    """
    Ofrece el catálogo de sales y devuelve el dict de la sal elegida.

    Opciones: una clave conocida del catálogo (ej. "KCl") o "custom" para
    ingresar los parámetros a mano. El dict devuelto tiene SIEMPRE la misma
    forma — nombre, cation, anion, soluble, Kps_M2 — venga del catálogo o sea
    custom, de modo que el resto del solver no distingue su origen.
    """
    nombres = list(catalogo.keys())
    print(f"    Available salts: {', '.join(nombres)}")
    print(f"    Or type 'custom' to enter a salt by hand.")
    entrada = input(f"  Which salt? [Enter={default_nombre}]: ").strip()
    if not entrada:
        entrada = default_nombre

    if entrada in catalogo:
        return catalogo[entrada]

    if entrada.lower() != "custom":
        print(f"  [WARNING] '{entrada}' is not in the catalog; enter it by hand.")

    # --- Sal custom: se pregunta todo y se arma el MISMO schema del catálogo ---
    nombre = input("    -> Salt name [Enter=custom]: ").strip() or "custom"
    z_p    = pedir_valor("    -> Cation valence z+", 1, int)
    z_m    = pedir_valor("    -> Anion valence z-", -1, int)
    D_p    = pedir_valor("    -> Cation D (m²/s)", 1.96e-9)
    D_m    = pedir_valor("    -> Anion D (m²/s)", 2.03e-9)
    r_sol  = input("    -> Is it fully soluble (does not precipitate)? [Enter=Yes / n=No]: ").strip().lower()
    soluble = r_sol not in ['n', 'no']
    Kps     = None if soluble else pedir_valor("    -> Ksp (M²)", 1.07e-2)
    return {
        "name":  nombre,
        "cation":  {"symbol": f"cation(z={z_p:+d})", "z": z_p, "D_m2s": D_p},
        "anion":   {"symbol": f"anion(z={z_m:+d})",  "z": z_m, "D_m2s": D_m},
        "soluble": soluble,
        "Ksp_M2":  Kps,
    }


# =============================================================================
# SELECCIÓN DE MALLA (explorador gráfico + historial, verbatim del original)
# =============================================================================
def seleccionar_malla_gui(default_dir):
    """
    Abre un explorador de archivos para elegir la malla, apuntando por default
    a default_dir (RESULTS/meshes/). El usuario selecciona UN archivo de la
    malla (_limits.json o _domain.xdmf) y de ahí se derivan la carpeta
    (INPUT_DIR) y el nombre base (m_name, quitando el sufijo).

    Devuelve (INPUT_DIR, m_name), o None si el usuario cancela. Si el explorador
    no se puede abrir (sin entorno gráfico, tkinter ausente), levanta excepción
    y el caller cae a los prompts de texto.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("\nOpening file browser to choose the mesh...")
    inicio = default_dir if os.path.isdir(default_dir) else os.path.dirname(default_dir)
    ruta = filedialog.askopenfilename(
        title="Choose a mesh file (_domain.xdmf)",
        initialdir=inicio,
        filetypes=[("Mesh files", "*_domain.xdmf"),
                   ("All files", "*.*")],
    )
    root.destroy()

    if not ruta:
        return None

    return derivar_malla(ruta)


def guardar_historial_malla(input_dir, m_name):
    """Guarda la malla elegida (ruta del _limits.json) para ofrecerla la próxima vez."""
    try:
        os.makedirs(os.path.dirname(HISTORIAL_MALLA), exist_ok=True)
        with open(HISTORIAL_MALLA, "w", encoding="utf-8") as f:
            f.write(os.path.join(input_dir, f"{m_name}_limits.json"))
    except Exception:
        pass   # el historial es una comodidad; si falla, no es crítico


def leer_historial_malla():
    """
    Devuelve (INPUT_DIR, m_name) de la última malla usada si el archivo de
    historial existe y la malla sigue en disco; si no, None.
    """
    if not os.path.exists(HISTORIAL_MALLA):
        return None
    try:
        with open(HISTORIAL_MALLA, "r", encoding="utf-8") as f:
            ruta = f.read().strip()
    except Exception:
        return None
    if not ruta or not os.path.exists(ruta):
        return None
    return derivar_malla(ruta)


# =============================================================================
# FLUJO DE PREGUNTAS (Sección 5 del solver original) → cfg
# =============================================================================
M_NAME_DEFAULT = "conito-film"


def armar_config():
    """Corre todas las preguntas del solver original y devuelve el cfg que
    consume motor_pnp.resolver(). Mismos textos, mismo orden, mismos defaults."""

    # --- Selección de la malla: historial → explorador → prompts de texto ---
    INPUT_DIR = None
    m_name    = None

    # 1. ¿Reutilizar la última malla usada?
    _ultima = leer_historial_malla()
    if _ultima is not None:
        _dir_u, _name_u = _ultima
        print(f"\n[Last mesh used]: '{_name_u}'  (in {_dir_u})")
        _resp = input("Keep working on it? (y/n) [Enter = y]: ").strip().lower()
        if _resp in ('', 's', 'si', 'y'):
            INPUT_DIR, m_name = _dir_u, _name_u

    # 2. Si no se reutilizó, elegir una nueva (explorador → fallback a texto)
    if INPUT_DIR is None:
        _seleccion = None
        try:
            _seleccion = seleccionar_malla_gui(RESULTADOS_MALLAS)
        except Exception as e:
            print(f"  [NOTE] Could not open the file browser ({e}). Using text prompts.")

        if _seleccion is not None:
            INPUT_DIR, m_name = _seleccion
        else:
            m_name = input(f"\nMesh base name (Enter = '{M_NAME_DEFAULT}'): ").strip() or M_NAME_DEFAULT
            _default_input = os.path.join(RESULTADOS_MALLAS, m_name)
            _dir_input = input(f"Mesh folder (Enter = {_default_input}): ").strip()
            INPUT_DIR  = os.path.expanduser(_dir_input) if _dir_input else _default_input

    # 3. Guardar para la próxima vez
    guardar_historial_malla(INPUT_DIR, m_name)
    print(f"  Chosen mesh: '{m_name}'  (in {INPUT_DIR})")

    # -------------------------------------------------------------------------
    # GRUPO 1 — Condiciones del medio (rara vez cambian)
    # -------------------------------------------------------------------------
    print("\n--- [1] Medium conditions ---")
    print(f"    Temperature        T     = {T_DEFAULT_K} K")
    print(f"    Dielectric const.  eps_r = {EPS_R_DEFAULT}")
    if preguntar_ok("  Use these values?"):
        T     = T_DEFAULT_K
        eps_r = EPS_R_DEFAULT
    else:
        T     = pedir_valor("  -> Temperature (K)", T_DEFAULT_K)
        eps_r = pedir_valor("  -> Dielectric constant", EPS_R_DEFAULT)

    # -------------------------------------------------------------------------
    # GRUPO 2 — Electrolito (sal del catálogo + concentración)
    # -------------------------------------------------------------------------
    print("\n--- [2] Electrolyte ---")
    catalogo_sales = cargar_catalogo_sales()
    sal = elegir_sal(catalogo_sales, SAL_DEFAULT)

    z_p, z_m = sal["cation"]["z"],     sal["anion"]["z"]
    D_p, D_m = sal["cation"]["D_m2s"], sal["anion"]["D_m2s"]

    if sal["soluble"]:
        sol_txt = "fully soluble (does not precipitate)"
    else:
        sol_txt = f"partially soluble (Ksp = {sal['Ksp_M2']:.2e} M²)"
    print(f"    Salt: {sal['name']}  →  "
          f"{sal['cation']['symbol']}(z={z_p:+d}, D={D_p:.2e})  "
          f"{sal['anion']['symbol']}(z={z_m:+d}, D={D_m:.2e})")
    print(f"    {sol_txt}")

    c0_mM = pedir_valor("  -> Bulk concentration (mM)", C0_DEFAULT_MM)

    # Concentración a cada lado. Hoy SIMÉTRICA (mismo valor en inlet y outlet).
    # ── Gradiente de concentración: NO IMPLEMENTADO (rama preparada, inerte) ──
    quiere_gradiente = input(
        "  Different concentration on each side (gradient)? [Enter=No]: ").strip().lower()
    if quiere_gradiente in ['s', 'si', 'y', 'yes']:
        print("  [NOTE] The concentration gradient is not implemented yet.")
        print("         A symmetric concentration (equal on both sides) will be used.")
        # (rama futura: pedir c0_inlet_mM y c0_outlet_mM por separado y rampar Δc)

    # -------------------------------------------------------------------------
    # GRUPO 3 — Carga superficial del canal (σ)  [importante: se pregunta siempre]
    # -------------------------------------------------------------------------
    print("\n--- [3] Channel surface charge ---")
    print(f"    σ = {SIGMA_DEFAULT_E_NM2} e/nm²  =  {SIGMA_DEFAULT_CM2:.4f} C/m²")
    if preguntar_ok("  Use this surface charge?"):
        sigma_Cm2 = SIGMA_DEFAULT_CM2
    else:
        sigma_e_nm2 = pedir_valor("  -> Surface charge (e/nm²)", SIGMA_DEFAULT_E_NM2)
        sigma_Cm2   = sigma_e_nm2 * E_CHARGE / (1e-9)**2
        print(f"      → σ = {sigma_Cm2:.4f} C/m²")

    aplicar_carga_coronas = preguntar_ok("  Apply this charge on the rings too (if any)?")

    # -------------------------------------------------------------------------
    # GRUPO 4 — Film(s): leer el JSON del mallador y preguntar el tipo de cada
    #           film presente. El JSON es la fuente de verdad sobre qué films hay.
    # -------------------------------------------------------------------------
    json_mallador = f"{INPUT_DIR}/{m_name}_limits.json"
    if not os.path.exists(json_mallador):
        print(f"\n[ERROR] Not found: {json_mallador}")
        print("        The solver requires the _limits.json generated by the mesher.")
        sys.exit(1)

    with open(json_mallador, "r") as f:
        meta_geo = json.load(f)

    lados_con_film = [lado for lado in ("tip", "base")
                      if meta_geo.get(f"include_film_{lado}", False)]

    films_cfg = []
    if not lados_con_film:
        print("\n--- [4] Films ---")
        print("    Geometry with NO film.")
    else:
        n_f = len(lados_con_film)
        print(f"\n--- [4] Charged films ({n_f} detected) ---")
        for lado in lados_con_film:
            tipo, n_e = elegir_tipo_film(lado)
            films_cfg.append({"side": lado, "type": tipo, "n_e_per_nm3": n_e})

        # Orden de carga (rampa secuencial). Solo se pregunta si hay 2+ films;
        # con uno solo el orden es trivial. Default: tip antes que base.
        if n_f >= 2:
            nombres = [f["side"] for f in films_cfg]
            print(f"\n  Film ramp order (they are charged sequentially):")
            print(f"    Available films: {', '.join(nombres)}")
            entrada = input(
                f"  Which one to charge first? [Enter={nombres[0]}]: ").strip().lower()
            if entrada in nombres and entrada != nombres[0]:
                films_cfg.sort(key=lambda f: 0 if f["side"] == entrada else 1)
                print(f"  → Ramp order: {', '.join(f['side'] for f in films_cfg)}")

    # -------------------------------------------------------------------------
    # GRUPO 5 — Barrido de voltaje
    # -------------------------------------------------------------------------
    print("\n--- [5] Voltage sweep ---")
    V_max_V = pedir_valor("  -> Maximum voltage (V)", V_MAX_DEFAULT_V)
    n_steps = pedir_valor("  -> Points per branch (including 0V)", N_STEPS_DEFAULT, int)
    _r_sol = input("  Save the solution in the .h5 for all voltages? (y = all) [Enter = only 0.1 V, 0.2 V...]: ").strip().lower()
    guardar_todas_sol = _r_sol in ('s', 'si', 'y', 'yes')

    # -------------------------------------------------------------------------
    # GRUPO 6 — Carpeta de salida (SIEMPRE dentro de RESULTS/solutions/)
    # -------------------------------------------------------------------------
    _sol_subdir  = f"{m_name}_{sal['name']}_{c0_mM}mM"
    _nombre_sol = input(f"\nOutput subfolder name (Enter = '{_sol_subdir}'): ").strip()
    if not _nombre_sol:
        _nombre_sol = _sol_subdir
    OUTPUT_DIR = os.path.join(RESULTADOS_SOLUCIONES, _nombre_sol)
    print(f"  → Output in: {OUTPUT_DIR}")

    print("\n>>> Parameter summary:")
    print(f"    Electrolyte:   {sal['name']}  ({c0_mM} mM)")
    print(f"    Surf. charge:  {sigma_Cm2:.4f} C/m²" + (" (with rings)" if aplicar_carga_coronas else " (wall only)"))
    print(f"    Max voltage:   ±{V_max_V} V  ({(n_steps - 1)*2 + 1} total points)")
    print(f"    T: {T} K  |  eps_r: {eps_r}")
    print("="*60 + "\n")

    return {
        "input_dir":  INPUT_DIR,
        "m_name":     m_name,
        "output_dir": OUTPUT_DIR,
        "T": T, "eps_r": eps_r,
        "salt": sal,
        "c0_mM": c0_mM,
        "sigma_Cm2": sigma_Cm2,
        "apply_charge_rings": aplicar_carga_coronas,
        "films": films_cfg,
        "V_max_V": V_max_V,
        "n_steps": n_steps,
        "guardar_todas_sol": guardar_todas_sol,
        # comportamiento original: si hay checkpoint compatible, PREGUNTA
        "reuse_equilibrium": "ask",
    }
