# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MESHER (Pillar 1). Two usage modes.
═══════════════════════════════════════════════════════════════════════════

  python launch_mallador.py                 → INTERACTIVE: opens the GUI (Streamlit)
                                       in the browser to draw the channel by hand.

  python launch_mallador.py geom.json       → BATCH: generates the mesh described in
                                       the JSON, asking nothing. Iterable.

The JSON describes the SAME geometry the GUI assembles (the fields of the
`Params` dataclass in Layer 1), plus the name and the output folder:

    {
      "name": "conico_demo",
      "output": "meshes/conico_demo",       # relative to cwd, or absolute
      "params": { ...Params fields... }      # see examples/
    }

Both modes end up calling the SAME `mallar(p, base_path, name)` function of
Layer 4, so the generated mesh is identical to the GUI's for the same
parameters. Output: <name>_domain.xdmf + _facets.xdmf + _limits.json
(+ .msh and _mesh.png), the contract read by the Solver (Pillar 2).

Requires gmsh + meshio + numpy + matplotlib (NO FEniCS).
"""
import os
import sys
import json

AQUI = os.path.dirname(os.path.abspath(__file__))


def _generar_batch(ruta_json):
    """Lee geom.json, arma el Params y llama a mallar() sin interacción."""
    if not os.path.isfile(ruta_json):
        # Error típico: el archivo vive en examples/ pero se llamó sin la carpeta.
        pista = ""
        alt = os.path.join(AQUI, "examples", os.path.basename(ruta_json))
        if os.path.isfile(alt):
            pista = (f"\n   It's in examples/. Try:  "
                     f"python launch_mallador.py examples/{os.path.basename(ruta_json)}")
        sys.exit(f"❌ Could not find '{ruta_json}'  (cwd: {os.getcwd()}).{pista}")

    try:
        with open(ruta_json, encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ The JSON '{ruta_json}' is malformed: {e}\n"
                 f"   Check that line/column. Typical causes: an extra or missing comma, "
                 f"an unclosed quote, or a TAB/newline inside a value.")

    for clave in ("name", "params"):
        if clave not in cfg:
            sys.exit(f"❌ The JSON '{ruta_json}' is missing the required field '{clave}'.")

    sys.path.insert(0, AQUI)
    from capa1_modelo import Params
    from capa4_malla import mallar

    # Construir el Params validando los nombres de campo (falla claro si sobra
    # o falta una clave, en vez de un TypeError críptico).
    validos = set(Params.__dataclass_fields__)
    desconocidos = set(cfg["params"]) - validos
    if desconocidos:
        sys.exit(f"❌ Unrecognized fields in 'params': {sorted(desconocidos)}\n"
                 f"   Valid: {sorted(validos)}")
    p = Params(**cfg["params"])

    nombre = cfg["name"]
    salida = cfg.get("output", os.path.join("meshes", nombre))
    if not os.path.isabs(salida):
        salida = os.path.abspath(salida)  # relativa al cwd desde donde se lanza

    print(f">>> Generating mesh '{nombre}'  →  {salida}")
    info = mallar(p, salida, nombre)
    print(f"✓ Mesh generated.  triangles={info['n_tri']}  segments={info['n_lin']}")
    print(f"  files: {nombre}_domain.xdmf / _facets.xdmf / _limits.json / .msh / _mesh.png")


def _abrir_gui():
    """Lanza la interfaz web (Streamlit) del dibujador."""
    import subprocess
    app = os.path.join(AQUI, "gui_app.py")
    print(">>> Opening the interactive mesher (Streamlit) in the browser...\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app], cwd=AQUI)


# Descripción de cada campo de 'params' (texto de ayuda; los nombres y
# defaults NO viven acá: se leen del dataclass Params, así no se desactualizan).
_DESCRIPCION = {
    "L_pore":            "channel length [m]",
    "D_tip":             "diameter of the tip mouth (the narrow one) [m]",
    "D_base":            "diameter of the base mouth (the wide one) [m]",
    "L_res":             "length of each reservoir [m]",
    "R_res":             "radius of the reservoirs [m]",
    "L_charge":          "width of the chargeable ring on each membrane face [m]",
    "L_far":             "width of the transition zone past the ring [m]",
    "include_film_tip":  "true/false: add a film attached to the tip mouth",
    "delta_film_tip":    "tip film thickness [m] (only if include_film_tip)",
    "include_film_base": "true/false: add a film attached to the base mouth",
    "delta_film_base":   "base film thickness [m] (only if include_film_base)",
    "N_PTS_WALL":        "points of the wall spline (default is usually enough)",
    "channel_type":        '"cylinder" | "conical" | "bullet" (wall profile)',
    "h_param":           "bullet profile scale [m]; ONLY used if channel_type=bullet",
}


def _mostrar_ayuda():
    """Imprime la guía del geom.json, con nombres y defaults leídos de Params."""
    sys.path.insert(0, AQUI)
    from capa1_modelo import Params
    print(__doc__)
    print("─" * 75)
    print("  \"params\" FIELDS (SI units: METERS; notation 20e-9 = 20 nm)")
    print("─" * 75)
    for nombre, campo in Params.__dataclass_fields__.items():
        desc = _DESCRIPCION.get(nombre, "")
        default = campo.default
        if isinstance(default, bool):
            d = "true" if default else "false"
        elif isinstance(default, float):
            d = f"{default:g}"
        else:
            d = str(default)
        print(f"  {nombre:<18} [default: {d:>8}]  {desc}")
    print("""
  All fields are OPTIONAL: whatever is missing takes its default.
  Fields with an unknown name abort with an error (protects against typos).

  BULLET channel (the paper's exponential profile):
      "params": { ..., "channel_type": "bullet", "h_param": 2.0e-6 }
  R(x) = R_base − (R_base − R_tip)·exp(−x/h_param), x measured from the tip mouth.
  small h ⇒ opens up fast near the tip; large h ⇒ smooth transition.

  Full ready-to-run examples in:  examples/
""")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) >= 2 else None
    if arg and arg.endswith(".json"):
        _generar_batch(arg)
    elif arg in ("-h", "--help", "help", "fields", "--fields"):
        _mostrar_ayuda()
    elif arg:
        sys.exit(f"❌ Unrecognized argument: '{arg}'.\n"
                 f"   Usage:  python launch_mallador.py [geom.json]     (no argument → GUI)\n"
                 f"           python launch_mallador.py help            (geom.json guide)")
    else:
        _abrir_gui()
