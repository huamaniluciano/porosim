# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MALLADOR · CAPA 4: MALLA (mesher multi-escala + export al solver)
  Implementación de las reglas de refinamiento físico para EDL y Bulk.
  Versión matemáticamente pura y delimitada en 2D para evitar lc <= 0.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import h5py
except Exception:
    pass

import numpy as np
import gmsh
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capa1_modelo import Params, TAGS
from capa3_gmsh import emitir_geometria


# =================================================================
# 1) GENERAR LA MALLA REFINADA MEDIANTE ORQUESTACIÓN DE FIELDS
# =================================================================
def _generar_malla_bruta(p: Params):
    from capa1_modelo import construir_estaciones, construir_slabs, construir_lineas
    
    estaciones = construir_estaciones(p)
    slabs = construir_slabs(estaciones, p)
    lineas = construir_lineas(estaciones, slabs, p)
    gtag_linea = {l.nombre: i + 1 for i, l in enumerate(lineas)}
    
    campos_a_orquestar = []
    lc_bulk = p.R_res / 5.0

    # Muestreo de los campos Distance: nº de puntos con que Gmsh discretiza cada
    # curva para MEDIR distancias. El default (~20) es muy bajo para curvas largas:
    # la pared mide ~L_pore, así que con 20 muestras quedan huecos de varios nm
    # entre ellas y un punto que está SOBRE la pared, pero entre dos muestras, se
    # mide a distancia > 0. El Threshold le asigna entonces un tamaño grueso y
    # aparecen los "abanicos" de elementos grandes colgando de la pared (la pared
    # nunca recibe su LcMin de 0.2 nm). Se fija un paso de muestreo ~0.2 nm.
    SAMP_STEP = 0.2e-9
    n_samp_wall = max(400, int(p.L_pore / SAMP_STEP))
    n_samp_memb = max(400, int(p.R_res  / SAMP_STEP))

    # ─────────────────────────────────────────────────────────────────────────
    # REGLA 1: Interior del Poro (Pared Fija y Eje Dinámico con MathEval Puro)
    # ─────────────────────────────────────────────────────────────────────────
    lc_tip_eje = 0.10 * p.R_tip
    lc_base_eje = 0.25 * p.R_base
    
    # R(x): perfil de la pared en sintaxis MathEval de Gmsh. DEBE coincidir con
    # R_canal de Capa 1 (misma fórmula) para que el refinamiento siga la pared real.
    if getattr(p, "channel_type", "conical") == "bullet":
        # R(x) = R_base - (R_base - R_tip) * exp(-(x - z_tip)/h_param)
        r_x = f"({p.R_base} - ({p.R_base} - {p.R_tip}) * exp(-(x - {p.z_tip}) / {p.h_param}))"
    else:
        r_x = f"({p.R_tip} + ({p.R_base} - {p.R_tip}) * (x - {p.z_tip}) / {p.L_pore})"
    # valor radial seguro para evitar división por cero
    r_safe = f"(sqrt(({r_x})*({r_x})) + 1e-15)"
    y_safe = f"sqrt(y*y)"
    
    # Lc en el eje a lo largo de Z
    lc_axis = f"({lc_tip_eje} + ({lc_base_eje} - {lc_tip_eje}) * (x - {p.z_tip}) / {p.L_pore})"
    
    # Fórmula estricta para el interior del poro
    math_pore = f"(0.2e-9 + ({lc_axis} - 0.2e-9) * (1.0 - {y_safe} / {r_safe}))"

    # --- HACK MATEMÁTICO: Step Functions sin operadores lógicos ---
    def step_up(val):
        return f"(0.5 * (1.0 + ({val}) / (sqrt(({val})*({val})) + 1e-15)))"
        
    def step_down(val):
        return f"(0.5 * (1.0 - ({val}) / (sqrt(({val})*({val})) + 1e-15)))"

    # Delimitadores axiales
    s_tip = step_up(f"x - {p.z_tip}")
    s_base = step_down(f"x - {p.z_base}")
    
    # NUEVO: Delimitador radial (vale 1 si estamos dentro del radio local del canal, 0 afuera)
    s_rad = step_up(f"{r_safe} - {y_safe}")
    
    # is_inside vale 1.0 estrictamente ADENTRO de la cavidad fluida
    is_inside = f"({s_tip} * {s_base} * {s_rad})"

    # Mezcla algebraica total: Si is_inside es 1, evalúa math_pore. Si es 0, evalúa lc_bulk.
    expr_final = f"({is_inside} * {math_pore} + (1.0 - {is_inside}) * {lc_bulk})"

    f_pore = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f_pore, "F", expr_final)
    campos_a_orquestar.append(f_pore)

    # Anclaje de la pared (0.2 nm estrictos sobre la curva física l_wall)
    if "l_wall" in gtag_linea:
        d_wall = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_wall, "CurvesList", [gtag_linea["l_wall"]])
        gmsh.model.mesh.field.setNumber(d_wall, "Sampling", n_samp_wall)

        t_wall = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_wall, "IField", d_wall)
        gmsh.model.mesh.field.setNumber(t_wall, "LcMin", 0.2e-9)
        gmsh.model.mesh.field.setNumber(t_wall, "LcMax", lc_bulk)
        gmsh.model.mesh.field.setNumber(t_wall, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(t_wall, "DistMax", p.R_tip)
        campos_a_orquestar.append(t_wall)

    # ─────────────────────────────────────────────────────────────────────────
    # REGLAS 2 y 3: Membranas (Zonas Sensibles y Lejanas)
    # ─────────────────────────────────────────────────────────────────────────
    lineas_membrana_fina = []
    lineas_membrana_outer = []
    
    for lado in ["tip", "base"]:
        for banda in ["aperture", "charge", "far"]:
            name = f"l_{lado}_{banda}"
            if name in gtag_linea:
                lineas_membrana_fina.append(gtag_linea[name])
        
        name_out = f"l_{lado}_outer"
        if name_out in gtag_linea:
            lineas_membrana_outer.append(gtag_linea[name_out])

    # Regla 2: Zonas próximas a la boca y corona (0.2 nm -> Bulk)
    if lineas_membrana_fina:
        d_memb_fina = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_memb_fina, "CurvesList", lineas_membrana_fina)
        gmsh.model.mesh.field.setNumber(d_memb_fina, "Sampling", n_samp_memb)

        t_memb_fina = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_memb_fina, "IField", d_memb_fina)
        gmsh.model.mesh.field.setNumber(t_memb_fina, "LcMin", 0.2e-9)
        gmsh.model.mesh.field.setNumber(t_memb_fina, "LcMax", lc_bulk)
        gmsh.model.mesh.field.setNumber(t_memb_fina, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(t_memb_fina, "DistMax", p.L_res)
        campos_a_orquestar.append(t_memb_fina)

    # Regla 3: Zona lejana 'outer' de la membrana (1.0 nm -> Bulk)
    if lineas_membrana_outer:
        d_memb_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_memb_outer, "CurvesList", lineas_membrana_outer)
        gmsh.model.mesh.field.setNumber(d_memb_outer, "Sampling", n_samp_memb)

        t_memb_outer = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_memb_outer, "IField", d_memb_outer)
        gmsh.model.mesh.field.setNumber(t_memb_outer, "LcMin", 1.0e-9)
        gmsh.model.mesh.field.setNumber(t_memb_outer, "LcMax", lc_bulk)
        gmsh.model.mesh.field.setNumber(t_memb_outer, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(t_memb_outer, "DistMax", p.L_res)
        campos_a_orquestar.append(t_memb_outer)

    # ─────────────────────────────────────────────────────────────────────────
    # REGLA 4: Interfaces Activas de los Films
    # ─────────────────────────────────────────────────────────────────────────
    lineas_films = []
    if "l_film_tip_face" in gtag_linea:
        lineas_films.append(gtag_linea["l_film_tip_face"])
    if "l_film_base_face" in gtag_linea:
        lineas_films.append(gtag_linea["l_film_base_face"])
        
    if lineas_films:
        d_films = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_films, "CurvesList", lineas_films)
        gmsh.model.mesh.field.setNumber(d_films, "Sampling", n_samp_memb)

        t_films = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_films, "IField", d_films)
        gmsh.model.mesh.field.setNumber(t_films, "LcMin", 0.2e-9)
        gmsh.model.mesh.field.setNumber(t_films, "LcMax", lc_bulk)
        gmsh.model.mesh.field.setNumber(t_films, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(t_films, "DistMax", p.L_res)
        campos_a_orquestar.append(t_films)

    # ─────────────────────────────────────────────────────────────────────────
    # REGLA 5: Orquestador Global (Operador Mínimo Semántico)
    # ─────────────────────────────────────────────────────────────────────────
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", campos_a_orquestar)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    
    gmsh.model.mesh.generate(2)


# =================================================================
# 2) EXPORTAR (Contrato del Solver - XDMF/JSON)
# =================================================================
def _exportar(p: Params, base_path, name):
    os.makedirs(base_path, exist_ok=True)

    msh_path = os.path.join(base_path, f"{name}.msh")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(msh_path)
    msh = meshio.read(msh_path)

    triangles = msh.get_cells_type("triangle")
    triangles_phys = msh.get_cell_data("gmsh:physical", "triangle")
    meshio.write(
        os.path.join(base_path, f"{name}_domain.xdmf"),
        meshio.Mesh(points=msh.points[:, :2], cells=[("triangle", triangles)], cell_data={"subdomains": [triangles_phys]}),
    )

    meshio.write(
        os.path.join(base_path, f"{name}_facets.xdmf"),
        meshio.Mesh(points=msh.points[:, :2], cells=[("line", msh.get_cells_type("line"))], cell_data={"f": [msh.get_cell_data("gmsh:physical", "line")]}),
    )

    tags_presentes = {
        "AXIS": TAGS["AXIS"], "WALL": TAGS["WALL"], "INLET": TAGS["INLET"], "OUTLET": TAGS["OUTLET"],
        "CHARGE_ZONE_TIP": TAGS["CHARGE_ZONE_TIP"], "CHARGE_ZONE_BASE": TAGS["CHARGE_ZONE_BASE"], "DOMAIN_FLUID": TAGS["DOMAIN_FLUID"],
    }
    if p.include_film_tip:
        tags_presentes["FILM_TIP_INTERFACE"] = TAGS["FILM_TIP_INTERFACE"]
        tags_presentes["DOMAIN_FILM_TIP"] = TAGS["DOMAIN_FILM_TIP"]
    if p.include_film_base:
        tags_presentes["FILM_BASE_INTERFACE"] = TAGS["FILM_BASE_INTERFACE"]
        tags_presentes["DOMAIN_FILM_BASE"] = TAGS["DOMAIN_FILM_BASE"]

    limites = {
        "m_name": name, "channel_type": str(getattr(p, "channel_type", "conical")),
        "z_tip": float(p.z_tip), "z_base": float(p.z_base), "R_tip": float(p.R_tip), "R_base": float(p.R_base),
        "L_pore": float(p.L_pore), "L_res": float(p.L_res), "R_res": float(p.R_res), "L_charge": float(p.L_charge),
        "L_far": float(p.L_far), "include_film_tip": bool(p.include_film_tip), "include_film_base": bool(p.include_film_base),
        "tags": tags_presentes,
    }
    if getattr(p, "channel_type", "conical") == "bullet": limites["h_param"] = float(p.h_param)
    if p.include_film_tip: limites.update({"delta_film_tip": float(p.delta_film_tip), "z_film_tip": float(p.z_film_tip)})
    if p.include_film_base: limites.update({"delta_film_base": float(p.delta_film_base), "z_film_base": float(p.z_film_base)})

    with open(os.path.join(base_path, f"{name}_limits.json"), "w") as f:
        json.dump(limites, f, indent=2)

    return msh, limites


# =================================================================
# 3) PNG DE LA MALLA (Inspección Visual)
# =================================================================
def _plot_malla(p: Params, msh, base_path, name):
    pts = msh.points[:, :2]
    tris = msh.get_cells_type("triangle")
    tphys = msh.get_cell_data("gmsh:physical", "triangle")

    color_de = {10: "#cfe8f7", 11: "#f6c5d4", 12: "#cfe9cf"}
    z, r = pts[:, 0] * 1e9, pts[:, 1] * 1e9

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 6))
    for ax, titulo in ((ax_full, "full view"), (ax_zoom, "tip zoom")):
        for tag in np.unique(tphys):
            m = tphys == tag
            ax.tripcolor(z, r, tris[m], facecolors=np.zeros(m.sum()), edgecolors="#33333333", linewidth=0.2, cmap=matplotlib.colors.ListedColormap([color_de.get(int(tag), "#dddddd")]))
            ax.tripcolor(z, -r, tris[m], facecolors=np.zeros(m.sum()), edgecolors="#33333333", linewidth=0.2, cmap=matplotlib.colors.ListedColormap([color_de.get(int(tag), "#dddddd")]))
        ax.set_aspect("equal")
        ax.set_xlabel("z (nm)"); ax.set_ylabel("r (nm)")
        ax.set_title(f"{name} — {titulo}")

    # Dynamic local zoom at the TIP
    z_film_tip = (p.z_tip - (p.delta_film_tip if p.include_film_tip else 0.0)) * 1e9
    z_tip_nm = p.z_tip * 1e9
    r_tip_nm = p.R_tip * 1e9
    L_charge_nm = getattr(p, "L_charge", 0.0) * 1e9
    
    # Adaptive margin: properly cover the charge corona and tip radius, with a reasonable minimum
    margin_x = max(150.0, r_tip_nm * 3.0, L_charge_nm * 1.5)
    
    ax_zoom.set_xlim(z_film_tip - margin_x * 0.5, z_tip_nm + margin_x)
    
    # In Y, cover tip radius + charge zone + adaptive margin
    margin_y = max(50.0, r_tip_nm * 0.5)
    r_max = r_tip_nm + L_charge_nm + margin_y

    ax_zoom.set_ylim(-r_max, r_max)

    fig.tight_layout()
    png_path = os.path.join(base_path, f"{name}_mesh.png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return png_path


# =================================================================
# 4) ORQUESTACIÓN DEL PIPELINE
# =================================================================
def mallar(p: Params, base_path, name):
    emitir_geometria(p)
    _generar_malla_bruta(p)
    msh, limites = _exportar(p, base_path, name)
    png = _plot_malla(p, msh, base_path, name)
    n_tri = len(msh.get_cells_type("triangle"))
    n_lin = len(msh.get_cells_type("line"))

    if gmsh.isInitialized():
        gmsh.finalize()
    return {"n_tri": n_tri, "n_lin": n_lin, "tags": list(limites["tags"].values()), "png": png}

if __name__ == "__main__":
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(_REPO, "RESULTS", "meshes")
    casos = [
        ("sin_film",  Params(include_film_tip=False, include_film_base=False)),
        ("film_tip",  Params(include_film_tip=True,  include_film_base=False)),
        ("film_base", Params(include_film_tip=False, include_film_base=True)),
        ("dos_films", Params(include_film_tip=True,  include_film_base=True)),
    ]
    print("=" * 70)
    print("  MESHER AVANZADO — Generando mallas físicas continuas")
    print(f"  Salida en: {OUT}")
    print("=" * 70)
    for name, p in casos:
        try:
            info = mallar(p, OUT, name)
            print(f"\n  [{name}]  ✓ Generación Exitosa")
            print(f"     Triángulos: {info['n_tri']} | Segmentos de Borde: {info['n_lin']}")
        except Exception as e:
            print(f"\n  [{name}]  ✗ FALLÓ LA GENERACIÓN → {e}")
            if gmsh.isInitialized():
                gmsh.finalize()
    print("\n  Proceso de Capa 4 finalizado con éxito.")