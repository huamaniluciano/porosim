# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — PNP SOLVER (Pillar 2). Two usage modes.
═══════════════════════════════════════════════════════════════════════════

  python solver.py                  → INTERACTIVE: the usual prompts
                                      (mesh, salt, σ, films, voltage).

  python solver.py params.json      → BATCH: runs the simulation described in
                                      the JSON, asking nothing. Iterable.

  python solver.py params.json --mesh PATH
                                    → HYBRID BATCH: the same physics from the
                                      JSON but on ANOTHER mesh (overrides the
                                      "mesh" field). Ideal for sweeping
                                      geometries with a single params.json.

  python solver.py help             → guide to the params.json fields.

The JSON references the mesher's mesh (Pillar 1) and defines the physics:

    {
      "mesh":  "path/to/<name>_domain.xdmf",     // or the mesh folder
      "output": "optional",                        // default RESULTS/solutions/...
      "params": { ...physical fields... }          // see 'help' and examples/
    }

Both modes end up calling the SAME resolver(cfg) function of motor_pnp.py
(the paper's physics, unchanged), so the simulation is identical for the same
parameters. Outputs: Solutions_*.h5 + IV_curve_*.txt + *_sim.json (the
contract read by the Extractor, Pillar 3).

Requires FEniCS (dolfin 2019.1.0) + numpy. Help and validation run without
FEniCS.
"""
import os
import sys
import json

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
# preguntas.py (modo interactivo de consola) vive en console_backup/
sys.path.insert(0, os.path.join(AQUI, "console_backup"))

from constantes import (
    E_CHARGE, E_NM3_TO_MOLAR, film_tipos,
    C0_DEFAULT_MM, V_MAX_DEFAULT_V, N_STEPS_DEFAULT, T_DEFAULT_K,
    EPS_R_DEFAULT, SIGMA_DEFAULT_E_NM2, SAL_DEFAULT, N_STEPS_FILM,
    RESULTADOS_SOLUCIONES, cargar_catalogo_sales, derivar_malla,
)

# Campos válidos de "params" con su default (None = obligatorio contextual).
# La ayuda se genera de esta tabla, así no puede desincronizarse del código.
CAMPOS_PARAMS = {
    "salt":                     (SAL_DEFAULT,        'salt from the sales.json catalog ("KCl", ...) or a complete custom dict'),
    "c0_mM":                   (C0_DEFAULT_MM,      "bulk concentration [mM], symmetric on both sides"),
    "c0_inlet_mM":             (None,               "optional; MUST equal c0_mM (gradient not implemented in v1)"),
    "c0_outlet_mM":            (None,               "optional; same as c0_inlet_mM"),
    "sigma_e_nm2":             (SIGMA_DEFAULT_E_NM2,"wall surface charge [e/nm²] (negative = negative wall)"),
    "sigma_on_rings":        (True,               "also apply σ on the membrane rings (if any)"),
    "T_K":                     (T_DEFAULT_K,        "temperature [K]"),
    "eps_r":                   (EPS_R_DEFAULT,      "relative dielectric constant of the medium"),
    "films":                   ([],                 'charge of each mesh film: [{"side":"tip","type":"4M"}] or {"side":..,"molar_charge":3.0}; order = ramp order'),
    "V_max_V":                 (V_MAX_DEFAULT_V,    "maximum sweep voltage [V] (sweeps ±V_max)"),
    "n_steps":                 (N_STEPS_DEFAULT,    "points per branch including 0V (more points = smoother ramp)"),
    "save_all_solutions":(False,              "true = save U in the .h5 at ALL voltages (default: only multiples of 0.1V)"),
    "n_steps_film":            (N_STEPS_FILM,       "steps of the ρ_film ramp (raise if charged film + dilute electrolyte does not converge)"),
    "reuse_equilibrium":       ("auto",             '"auto" = use a compatible checkpoint if it exists; false = always recompute'),
    # ── perillas numéricas avanzadas (casi nunca tocar; defaults = paper) ──
    "n_steps_sigma":           (10,                 "steps of the σ ramp (Stage 1); almost never touch"),
    "tol_sigma_abs":           (1e-20,              "Newton absolute tolerance in the σ ramp; almost never touch"),
    "tol_sigma_rel":           (1e-8,               "Newton relative tolerance in the σ ramp; almost never touch"),
    "tol_film_abs":            (1e-20,              "Newton absolute tolerance in the ρ_film ramp; almost never touch"),
    "tol_film_rel":            (1e-8,               "Newton relative tolerance in the ρ_film ramp; almost never touch"),
}


def _mostrar_ayuda():
    print(__doc__)
    print("─" * 77)
    print('  "params" FIELDS (all optional except "films" if the mesh has films)')
    print("─" * 77)
    for nombre, (default, desc) in CAMPOS_PARAMS.items():
        if default is None:
            d = "—"
        elif isinstance(default, bool):
            d = "true" if default else "false"
        elif isinstance(default, list):
            d = "[]"
        else:
            d = f"{default:g}" if isinstance(default, float) else str(default)
        print(f"  {nombre:<26} [default: {d:>6}]  {desc}")
    print(f"""
  Catalog salts: {', '.join(cargar_catalogo_sales().keys())}   (add one = edit sales.json)
  Catalog film types: {', '.join(film_tipos.keys())}   (or "molar_charge": <number in M>)

  "mesh" accepts the path to the <name>_domain.xdmf or to the mesh folder.
  "output": subfolder of RESULTS/solutions/ by default; a relative (to cwd)
            or absolute path replaces it.

  KNOWN LIMITATIONS (see README):
    · Concentration gradient: NOT implemented (c_inlet ≠ c_outlet → error).
    · Highly charged film + very dilute electrolyte: raise n_steps and/or
      n_steps_film (slower ramps) if it does not converge or the IV collapses.

  Full ready-to-run examples in:  examples/
""")


# =============================================================================
# MODO BATCH
# =============================================================================
def _err(msg):
    sys.exit(f"❌ {msg}")


def _resolver_malla(ruta_malla, ruta_json):
    """Acepta ruta a un archivo de la malla o a su carpeta; devuelve
    (input_dir, m_name) verificando que estén los 3 archivos del contrato."""
    ruta = os.path.expanduser(ruta_malla)
    if not os.path.isabs(ruta):
        ruta = os.path.abspath(ruta)   # relativa al cwd desde donde se lanza

    if os.path.isdir(ruta):
        limites = [f for f in sorted(os.listdir(ruta)) if f.endswith("_limits.json")]
        if len(limites) != 1:
            _err(f"'{ruta_malla}' has {len(limites)} *_limits.json files "
                 f"(expected exactly 1). Point to the file directly.")
        ruta = os.path.join(ruta, limites[0])
    elif not os.path.exists(ruta):
        _err(f"Could not find the mesh '{ruta_malla}' (cwd: {os.getcwd()}).\n"
             f"   Generate it with the mesher (Pillar 1) and point 'mesh' to the _domain.xdmf.")

    input_dir, m_name = derivar_malla(ruta)
    # Los .xdmf son formato doble: el XML liviano + su .h5 compañero con los
    # datos pesados. Sin el .h5, FEniCS muere con "Unable to open HDF5 file"
    # recién al cargar — mejor cortarlo acá con un mensaje claro.
    faltan = [f"{m_name}{suf}" for suf in ("_limits.json", "_domain.xdmf",
                                           "_facets.xdmf", "_domain.h5", "_facets.h5")
              if not os.path.exists(os.path.join(input_dir, f"{m_name}{suf}"))]
    if faltan:
        _err(f"The mesh '{m_name}' is missing contract files: {faltan}\n"
             f"   (the _domain/_facets .xdmf need their companion .h5 files; "
             f"copy the whole mesh folder)")
    return input_dir, m_name


def _resolver_sal(valor):
    """'KCl' → entrada del catálogo; dict → validar schema custom."""
    if isinstance(valor, str):
        catalogo = cargar_catalogo_sales()
        if valor not in catalogo:
            _err(f'Salt "{valor}" is not in the catalog. Available: '
                 f'{", ".join(catalogo.keys())} (or a custom dict, see help).')
        return catalogo[valor]
    if isinstance(valor, dict):
        requeridas = {"name", "cation", "anion", "soluble", "Ksp_M2"}
        faltan = requeridas - set(valor)
        if faltan:
            _err(f"The custom salt is missing keys: {sorted(faltan)}")
        for ion in ("cation", "anion"):
            if not {"z", "D_m2s"} <= set(valor[ion]):
                _err(f'The custom salt needs "{ion}": {{"z": ..., "D_m2s": ...}}')
            valor[ion].setdefault("symbol", f"{ion}(z={valor[ion]['z']:+d})")
        return valor
    _err('"salt" must be a catalog name (string) or a custom dict.')


def _resolver_films(films_json, meta_geo):
    """Valida las asignaciones de carga contra los films de la malla y
    devuelve la lista para cfg (orden del JSON = orden de rampa)."""
    lados_malla = [lado for lado in ("tip", "base")
                   if meta_geo.get(f"include_film_{lado}", False)]

    if not isinstance(films_json, list):
        _err('"films" must be a list: [{"side": "tip", "type": "4M"}, ...]')

    lados_json = []
    films_cfg = []
    for f in films_json:
        lado = f.get("side")
        if lado not in ("tip", "base"):
            _err(f'Film with invalid "side": {f!r} (must be "tip" or "base").')
        if lado in lados_json:
            _err(f'Film "{lado}" repeated in "films".')
        lados_json.append(lado)

        tiene_tipo  = "type" in f
        tiene_molar = "molar_charge" in f
        if tiene_tipo == tiene_molar:   # ninguno o ambos
            _err(f'Film "{lado}": give "type" (catalog: {", ".join(film_tipos)}) '
                 f'OR "molar_charge" (number in M), not both.')
        if tiene_tipo:
            if f["type"] not in film_tipos:
                _err(f'Film "{lado}": type "{f["type"]}" is not in the catalog '
                     f'({", ".join(film_tipos)}). For another value use "molar_charge".')
            tipo, n_e = f["type"], film_tipos[f["type"]]
        else:
            molar = float(f["molar_charge"])
            tipo, n_e = f"custom_{molar:.2f}M", molar / E_NM3_TO_MOLAR

        films_cfg.append({"side": lado, "type": tipo, "n_e_per_nm3": n_e})

    sobran = set(lados_json) - set(lados_malla)
    faltan = set(lados_malla) - set(lados_json)
    if sobran:
        _err(f"The JSON assigns charge to film(s) {sorted(sobran)} but the mesh does not have them "
             f"(mesh films: {lados_malla or 'none'}).")
    if faltan:
        _err(f"The mesh has film(s) {sorted(faltan)} with no charge assigned: "
             f'add them to "films" in the JSON.')
    return films_cfg


def _generar_batch(ruta_json, malla_override=None):
    """Lee params.json, valida todo y llama a resolver() sin interacción.
    Si malla_override viene (flag --mesh), pisa el campo "mesh" del JSON:
    misma física, otra geometría — un solo params.json sirve para barrer mallas."""
    if not os.path.isfile(ruta_json):
        pista = ""
        alt = os.path.join(AQUI, "examples", os.path.basename(ruta_json))
        if os.path.isfile(alt):
            pista = (f"\n   It's in examples/. Try:  "
                     f"python solver.py examples/{os.path.basename(ruta_json)}")
        _err(f"Could not find '{ruta_json}'  (cwd: {os.getcwd()}).{pista}")

    try:
        with open(ruta_json, encoding="utf-8") as f:
            spec = json.load(f)
    except json.JSONDecodeError as e:
        _err(f"The JSON '{ruta_json}' is malformed: {e}\n"
             f"   Check that line/column. Typical causes: an extra or missing comma, "
             f"an unclosed quote, or a TAB/newline inside a value.")

    if malla_override:
        spec["mesh"] = malla_override
        print(f"  [--mesh] JSON mesh overridden by: {malla_override}")
    if "mesh" not in spec:
        _err(f"The JSON '{ruta_json}' has no 'mesh' field "
             f"(add it to the JSON or pass it with:  --mesh PATH).")
    params = spec.get("params", {})

    # --- nombres de campo válidos (protege contra typos) ---
    desconocidos = set(params) - set(CAMPOS_PARAMS)
    if desconocidos:
        _err(f"Unrecognized fields in 'params': {sorted(desconocidos)}\n"
             f"   Valid: {sorted(CAMPOS_PARAMS)}   (python solver.py help)")

    # --- malla + su JSON de geometría ---
    input_dir, m_name = _resolver_malla(spec["mesh"], ruta_json)
    with open(os.path.join(input_dir, f"{m_name}_limits.json")) as f:
        meta_geo = json.load(f)

    # --- física (defaults de CAMPOS_PARAMS; tipos coercionados) ---
    def get(campo, cast=None):
        v = params.get(campo, CAMPOS_PARAMS[campo][0])
        return cast(v) if (cast and v is not None) else v

    c0_mM = get("c0_mM", float)

    # Gradiente: NO implementado en v1 — rechazar explícito y claro.
    for lado_c in ("c0_inlet_mM", "c0_outlet_mM"):
        v = params.get(lado_c)
        if v is not None and float(v) != c0_mM:
            _err(f"{lado_c} = {v} ≠ c0_mM = {c0_mM}: the concentration gradient "
                 f"is NOT implemented in v1 (see README, limitations). "
                 f"Use a symmetric concentration.")

    sal        = _resolver_sal(get("salt"))
    films_cfg  = _resolver_films(get("films"), meta_geo)
    sigma_Cm2  = get("sigma_e_nm2", float) * E_CHARGE / (1e-9)**2

    reusar = get("reuse_equilibrium")
    if reusar not in ("auto", True, False):
        _err('"reuse_equilibrium" must be "auto", true or false (batch does not ask).')

    # --- carpeta de salida ---
    salida = spec.get("output")
    if not salida:
        salida = os.path.join(RESULTADOS_SOLUCIONES, f"{m_name}_{sal['name']}_{c0_mM}mM")
    elif not os.path.isabs(salida):
        salida = os.path.abspath(salida)   # relativa al cwd

    cfg = {
        "input_dir":  input_dir,
        "m_name":     m_name,
        "output_dir": salida,
        "T":          get("T_K", float),
        "eps_r":      get("eps_r", float),
        "salt":        sal,
        "c0_mM":      c0_mM,
        "sigma_Cm2":  sigma_Cm2,
        "apply_charge_rings": bool(get("sigma_on_rings")),
        "films":      films_cfg,
        "V_max_V":    get("V_max_V", float),
        "n_steps":    get("n_steps", int),
        "guardar_todas_sol": bool(get("save_all_solutions")),
        "n_steps_film":      get("n_steps_film", int),
        "reuse_equilibrium": reusar,
        "n_steps_sigma":     get("n_steps_sigma", int),
        "tol_sigma_abs":     get("tol_sigma_abs", float),
        "tol_sigma_rel":     get("tol_sigma_rel", float),
        "tol_film_abs":      get("tol_film_abs", float),
        "tol_film_rel":      get("tol_film_rel", float),
    }

    print("\n" + "="*60)
    print("   POROSIM — PNP SOLVER  (batch mode)")
    print("="*60)
    print(f"    Mesh:          {m_name}  (in {input_dir})")
    print(f"    Electrolyte:   {sal['name']}  ({c0_mM} mM)")
    print(f"    Surf. charge:  {sigma_Cm2:.4f} C/m²"
          + (" (with rings)" if cfg["apply_charge_rings"] else " (wall only)"))
    if films_cfg:
        print(f"    Films:         " + ", ".join(f"{f['side']}={f['type']}" for f in films_cfg))
    print(f"    Max voltage:   ±{cfg['V_max_V']} V  ({(cfg['n_steps'] - 1)*2 + 1} total points)")
    print(f"    T: {cfg['T']} K  |  eps_r: {cfg['eps_r']}")
    print(f"    Output:        {salida}")
    print("="*60 + "\n")

    from motor_pnp import resolver
    resolver(cfg)


def _interactivo():
    print("\n" + "="*60)
    print("   POROSIM — PNP SOLVER")
    print("="*60)
    from preguntas import armar_config
    cfg = armar_config()
    from motor_pnp import resolver
    resolver(cfg)




if __name__ == "__main__":
    argv = sys.argv[1:]

    # Flag opcional --mesh RUTA (solo tiene sentido junto a un params.json)
    malla_override = None
    if "--mesh" in argv:
        i = argv.index("--mesh")
        if i + 1 >= len(argv):
            sys.exit("❌ --mesh needs a path:  python solver.py params.json --mesh PATH")
        malla_override = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    arg = argv[0] if argv else None
    if arg and arg.endswith(".json"):
        _generar_batch(arg, malla_override)
    elif malla_override:
        sys.exit("❌ --mesh only goes together with a params.json:\n"
                 "   python solver.py params.json --mesh PATH")
    elif arg in ("-h", "--help", "help", "fields", "--fields"):
        _mostrar_ayuda()
    elif arg:
        sys.exit(f"❌ Unrecognized argument: '{arg}'.\n"
                 f"   Usage:  python solver.py [params.json] [--mesh PATH]   (no argument → interactive)\n"
                 f"           python solver.py help                          (params.json guide)")
    else:
        _interactivo()
