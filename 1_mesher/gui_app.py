# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MESHER · GUI (Streamlit): channel + films + reservoirs
  Run with:   python launch_mallador.py    (or directly: streamlit run gui_app.py)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys

try:
    import h5py
except Exception:
    pass

import subprocess
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# El recuadro de aspecto fijo (adjustable="datalim") hace que matplotlib expanda
# el rango de datos para mantener 1:1; avisa con un UserWarning inofensivo en cada
# render. Lo silenciamos para no ensuciar la consola del dibujador.
warnings.filterwarnings("ignore", message="Ignoring fixed.*aspect.*adjustable")

from gui_dibujo import dibujar_canal, leyenda_handles, estado_demo


# ---------------------------------------------------------------------
# Abrir la malla en la GUI de Gmsh (proceso aparte, no bloqueante)
# ---------------------------------------------------------------------
# Funciona porque el backend de Streamlit corre en la máquina local (con
# DISPLAY disponible): el subprocess abre la ventana de Gmsh en el escritorio
# del usuario para inspeccionar/hacer zoom, sin bloquear la app.
# Usamos sys.executable (-c inline) en vez del binario bin/gmsh: ese wrapper
# tiene shebang "#!/usr/bin/env python" que puede resolver a otro intérprete
# sin el módulo gmsh; el inline garantiza el Python del entorno actual.
def abrir_en_gmsh(msh_path):
    code = "import sys, gmsh; gmsh.initialize(sys.argv, run=True); gmsh.finalize()"
    subprocess.Popen([sys.executable, "-c", code, msh_path],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


st.set_page_config(page_title="PoroSim — Geometry drawer", layout="wide")
st.title("PoroSim · Geometry drawer")
st.caption("Channel + films + reservoirs.")


# ---------------------------------------------------------------------
# Control sincronizado slider + number_input
# (patrón con keys auxiliares + callbacks, estable en esta versión de Streamlit)
# ---------------------------------------------------------------------
def control_valor(label, key, min_v, max_v, step, default):
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    if key not in st.session_state:
        st.session_state[key] = float(default)
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[key]
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[key]

    def desde_slider():
        v = st.session_state[slider_key]
        st.session_state[key] = v
        st.session_state[input_key] = v

    def desde_input():
        v = st.session_state[input_key]
        st.session_state[key] = v
        st.session_state[slider_key] = v

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(label, min_value=float(min_v), max_value=float(max_v),
                  step=float(step), key=slider_key, on_change=desde_slider)
    with c2:
        st.number_input(label, min_value=float(min_v), max_value=float(max_v),
                        step=float(step), key=input_key,
                        label_visibility="collapsed", on_change=desde_input)
    return float(st.session_state[key])


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:

    st.header("Channel")
    tipo = st.radio("Geometry type", ["cylinder", "conical", "bullet"], horizontal=True)

    if tipo == "cylinder":
        d = control_valor("Diameter (nm)", "d_cil", 1.0, 2000.0, 5.0, 10.0)
        d_tip_nm = d_base_nm = d
    else:
        d_tip_nm  = control_valor("Tip mouth diameter (nm)",  "d_tip", 1.0, 2000.0, 5.0, 10.0)
        d_base_nm = control_valor("Base mouth diameter (nm)", "d_base", 1.0, 2000.0, 5.0, 50.0)

    if tipo == "bullet":
        st.caption("Exponential profile R(x) = R_base − (R_base − R_tip)·exp(−x/h), "
                   "with x measured from the tip mouth. small h ⇒ opens up fast near "
                   "the tip; large h ⇒ smooth transition toward R_base.")
        h_param_nm = control_valor("Transition scale h (nm)", "h_param",
                                   500.0, 12000.0, 100.0, 1000.0)
    else:
        h_param_nm = None

    l_pore_nm = control_valor("Channel length (nm)", "l_pore", 50.0, 12000.0, 1.0, 100.0)

    st.divider()
    st.header("Charge zone (ring)")
    st.caption("Subdivision of the membrane face that the solver will use as the "
               "charged zone. Geometry only here: it sets where and how wide, not "
               "the sign or the value of the charge. Symmetric tip/base.")

    usar_corona = st.checkbox("Define a charge zone on the membrane",
                              value=False, key="ck_corona")
    if usar_corona:
        l_charge_nm = control_valor("Charged zone width L_charge (nm)", "l_charge",
                                    0.5, 1000.0, 0.5, 5.0)
        l_far_nm = control_valor("Transition width L_far (nm)", "l_far",
                                 0.5, 1000.0, 0.5, 5.0)
    else:
        l_charge_nm = None
        l_far_nm = None

    st.divider()
    st.header("Films")
    st.caption("Attached to the mouth (tip or base). The charge is defined later, "
               "in the solver; geometry only here.")

    usar_film_tip = st.checkbox("Film on the tip side", value=False, key="ck_film_tip")
    if usar_film_tip:
        delta_tip_nm = control_valor("Tip film width (nm)", "delta_tip", 1.0, 200.0, 0.5, 12.0)
    else:
        delta_tip_nm = None

    usar_film_base = st.checkbox("Film on the base side", value=False, key="ck_film_base")
    if usar_film_base:
        delta_base_nm = control_valor("Base film width (nm)", "delta_base", 1.0, 200.0, 0.5, 12.0)
    else:
        delta_base_nm = None

    st.divider()
    st.header("Reservoirs")
    st.caption("On each side of the channel. When enabled, the full view zooms "
               "out to show the real proportion. Without reservoirs, the geometry "
               "cannot be generated yet.")

    usar_res = st.checkbox("Add reservoirs", value=False, key="ck_res")
    if usar_res:
        l_res_nm = control_valor("Length of each reservoir L_res (nm)", "l_res",
                                 100.0, 15000.0, 100.0, 500.0)
        r_res_nm = control_valor("Reservoir radius R_res (nm)", "r_res",
                                 100.0, 15000.0, 100.0, 400.0)
    else:
        l_res_nm = None
        r_res_nm = None


# ---------------------------------------------------------------------
# Estado geométrico
# ---------------------------------------------------------------------
# Las etiquetas de la GUI ya coinciden con los valores de estado/Params.
tipo_estado = {"cylinder": "cylinder", "conical": "conical", "bullet": "bullet"}[tipo]
st_estado = estado_demo(
    type=tipo_estado,
    D_tip=d_tip_nm * 1e-9,
    D_base=d_base_nm * 1e-9,
    L_pore=l_pore_nm * 1e-9,
    L_charge=(l_charge_nm * 1e-9 if usar_corona else 0.0),
    film_tip=({"delta": delta_tip_nm * 1e-9} if usar_film_tip else None),
    film_base=({"delta": delta_base_nm * 1e-9} if usar_film_base else None),
    L_res=(l_res_nm * 1e-9 if usar_res else None),
    R_res=(r_res_nm * 1e-9 if usar_res else None),
    h_param=(h_param_nm * 1e-9 if tipo == "bullet" else None),
)
# L_far no es parte del estado_demo base; lo anexamos para el dibujo/tags.
st_estado["L_far"] = (l_far_nm * 1e-9 if usar_corona else 0.0)
st_estado["usar_corona"] = usar_corona

# Validaciones geométricas (avisos, no rompen). geometria_valida marca si se
# puede generar. Política: avisar sin tocar los valores.
avisos = []
geometria_valida = True

if usar_res:
    if r_res_nm <= max(d_tip_nm, d_base_nm) / 2.0:
        avisos.append("R_res should be larger than the radius of the biggest mouth.")
        geometria_valida = False
    if usar_film_tip and delta_tip_nm >= l_res_nm:
        avisos.append("The tip film is wider than the reservoir (delta_tip ≥ L_res).")
        geometria_valida = False
    if usar_film_base and delta_base_nm >= l_res_nm:
        avisos.append("The base film is wider than the reservoir (delta_base ≥ L_res).")
        geometria_valida = False

# Corona: restricciones sobre la cara MÁS CORTA (la del radio de boca mayor),
# con margen del 5% reservado para el segmento 'outer'. Todo en nm.
if usar_corona:
    if not usar_res:
        avisos.append("The charge zone needs the reservoirs defined "
                      "(the limit depends on R_res).")
        geometria_valida = False
    else:
        r_max_nm = max(d_tip_nm, d_base_nm) / 2.0
        cara_nm = r_res_nm - r_max_nm                # espacio radial de la cara más corta
        if cara_nm <= 0:
            avisos.append("No membrane face: R_res ≤ radius of the biggest mouth.")
            geometria_valida = False
        else:
            tope = cara_nm * 0.95                     # 5% de margen para 'outer'
            if l_far_nm < l_charge_nm:
                avisos.append(f"L_far ({l_far_nm} nm) must be ≥ L_charge "
                              f"({l_charge_nm} nm).")
                geometria_valida = False
            if l_charge_nm > tope:
                avisos.append(f"L_charge ({l_charge_nm} nm) exceeds the face: max "
                              f"≈ {tope:.1f} nm (leaves 5% for 'outer').")
                geometria_valida = False
            if l_charge_nm + l_far_nm > tope:
                avisos.append(f"L_charge + L_far ({l_charge_nm + l_far_nm} nm) exceeds "
                              f"the face: max ≈ {tope:.1f} nm (leaves 5% for 'outer').")
                geometria_valida = False


# ---------------------------------------------------------------------
# Dibujos
# ---------------------------------------------------------------------
# La vista completa (zoom-out con reservorios) solo aparece cuando hay
# reservorios definidos; antes de eso no hay proporción real que mostrar.
if usar_res:
    col_zoom, col_full = st.columns(2)
else:
    col_zoom, col_full = st.container(), None

with col_zoom:
    st.subheader("Channel zoom")
    fig_z, ax_z = plt.subplots(figsize=(6, 5))
    dibujar_canal(ax_z, st_estado, vista="zoom")
    st.pyplot(fig_z)
    plt.close(fig_z)

if col_full is not None:
    with col_full:
        st.subheader("Full view")
        fig_c, ax_c = plt.subplots(figsize=(6, 5))
        dibujar_canal(ax_c, st_estado, vista="completa")
        ax_c.legend(handles=leyenda_handles(st_estado), loc="upper right",
                    fontsize=7, framealpha=0.9)
        st.pyplot(fig_c)
        plt.close(fig_c)

for a in avisos:
    st.warning(a)


# ---------------------------------------------------------------------
# Estado actual + estado de "generable"
# ---------------------------------------------------------------------
with st.expander("Current state (what defines the geometry)"):
    films_txt = []
    if usar_film_tip:
        films_txt.append(f"tip (width {delta_tip_nm} nm)")
    if usar_film_base:
        films_txt.append(f"base (width {delta_base_nm} nm)")
    st.write({
        "type": st_estado["type"],
        "h (nm)": (round(h_param_nm, 3) if tipo == "bullet" else "n/a"),
        "D_tip (nm)":  round(st_estado["D_tip"]  * 1e9, 3),
        "D_base (nm)": round(st_estado["D_base"] * 1e9, 3),
        "L_pore (nm)": round(st_estado["L_pore"] * 1e9, 3),
        "ring": (f"L_charge {l_charge_nm} nm, L_far {l_far_nm} nm"
                 if usar_corona else "not set"),
        "films": ", ".join(films_txt) if films_txt else "none",
        "L_res (nm)": (round(l_res_nm, 3) if usar_res else "not set"),
        "R_res (nm)": (round(r_res_nm, 3) if usar_res else "not set"),
    })
    st.caption("The cylinder carries D_tip = D_base internally; the topology and "
               "the tags are the same as for the conical one.")

st.divider()
st.header("Generate geometry")

if not usar_res:
    st.info("The reservoirs are missing, needed to generate the mesh.")
elif not geometria_valida:
    st.error("Invalid geometry: check the warnings above before generating.")
else:
    st.success("Geometry complete and valid.")
    nombre = st.text_input("Mesh name", value="drawn", key="nombre_malla")

    # Carpeta fija: <repo>/RESULTS/meshes/<nombre>. El repo es el padre de la
    # carpeta del dibujador (1.Mallador/).
    aqui = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(aqui)
    base_path = os.path.join(repo, "RESULTS", "meshes", nombre)
    msh_path = os.path.join(base_path, f"{nombre}.msh")

    abrir_gmsh = st.checkbox(
        "Open in Gmsh after generating (separate window to zoom/inspect)",
        value=True, key="ck_abrir_gmsh")

    col_gen, col_open = st.columns([1, 1])
    with col_gen:
        generar = st.button("Generate geometry", type="primary")
    with col_open:
        reabrir = st.button("🔍 Open mesh in Gmsh",
                            disabled=not os.path.exists(msh_path),
                            help="Reopens the last mesh generated with this name, "
                                 "without re-meshing.")

    if generar:
        from gui_a_params import estado_a_params
        from capa4_malla import mallar

        try:
            with st.spinner("Generating mesh (Layer 4)…"):
                p = estado_a_params(st_estado)
                info = mallar(p, base_path, nombre)
            st.success(f"Mesh generated in: {base_path}")
            st.write({
                "triangles": info["n_tri"],
                "facets": info["n_lin"],
                "tags": sorted(info["tags"]),
                "files": [f"{nombre}.msh", f"{nombre}_domain.xdmf",
                          f"{nombre}_facets.xdmf", f"{nombre}_limits.json",
                          f"{nombre}_mesh.png"],
            })
            png = os.path.join(base_path, f"{nombre}_mesh.png")
            if os.path.exists(png):
                st.image(png, caption="Proto-mesh (coarse refinement)")
            if abrir_gmsh:
                try:
                    abrir_en_gmsh(msh_path)
                    st.info("Opening Gmsh in a separate window… (may take "
                            "a few seconds to appear).")
                except Exception as e:
                    st.warning(f"Could not open Gmsh automatically: {e}")
        except Exception as e:
            st.error(f"Generation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    if reabrir:
        try:
            abrir_en_gmsh(msh_path)
            st.info("Opening Gmsh in a separate window…")
        except Exception as e:
            st.warning(f"Could not open Gmsh: {e}")