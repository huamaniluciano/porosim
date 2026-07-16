# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — SOLVER · GUI Streamlit
  Se corre con:   python solver.py
  (o directo: streamlit run gui_app.py)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import glob
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Directorio de este script
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

# Imports del solver
from constantes import (
    RESULTADOS_MALLAS, RESULTADOS_SOLUCIONES, C0_DEFAULT_MM,
    V_MAX_DEFAULT_V, N_STEPS_DEFAULT, T_DEFAULT_K, EPS_R_DEFAULT,
    SIGMA_DEFAULT_E_NM2, E_CHARGE, E_NM3_TO_MOLAR, N_STEPS_FILM, SAL_DEFAULT,
    cargar_catalogo_sales, film_tipos, EQ_DIR, buscar_checkpoint_compatible
)
from gui_dibujo_solver import dibujar_canal_fisica, leyenda_handles_fisica, dibujar_proto_plot

st.set_page_config(page_title="PoroSim — PNP Solver", layout="wide")
st.title("PoroSim · PNP Solver")
st.caption("Physical setup, visualization of the charged domain and voltage sweep.")


# ---------------------------------------------------------------------
# Escaneo de Mallas Disponibles (verificación del contrato)
# ---------------------------------------------------------------------
def buscar_mallas():
    mallas = []
    if not os.path.exists(RESULTADOS_MALLAS):
        return mallas
    # Buscar límites
    limites_files = glob.glob(os.path.join(RESULTADOS_MALLAS, "**", "*_limits.json"), recursive=True)
    for lf in limites_files:
        base_name = os.path.basename(lf)[:-len("_limits.json")]
        root = os.path.dirname(lf)
        # Comprobar el resto de archivos del contrato
        if (os.path.exists(os.path.join(root, f"{base_name}_domain.xdmf")) and 
            os.path.exists(os.path.join(root, f"{base_name}_facets.xdmf"))):
            mallas.append({
                "name": base_name,
                "ruta_json": lf,
                "input_dir": root
            })
    return sorted(mallas, key=lambda x: x["name"])


mallas_disponibles = buscar_mallas()


if "ruta_manual_seleccionada" not in st.session_state:
    st.session_state["ruta_manual_seleccionada"] = ""

def seleccionar_malla_local():
    """Abre un diálogo tkinter nativo para buscar el archivo _limits.json."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Buscar el punto de partida (workspace o raíz del proyecto)
        from constantes import REPO_ROOT
        workspace_dir = os.path.dirname(REPO_ROOT)
        if not os.path.exists(workspace_dir):
            workspace_dir = REPO_ROOT
            
        ruta = filedialog.askopenfilename(
            title="Choose the mesh's _domain.xdmf file",
            initialdir=workspace_dir,
            filetypes=[("Mesh files", "*_domain.xdmf"),
                       ("All files", "*.*")]
        )
        root.destroy()
        return ruta
    except Exception as e:
        return f"ERROR: {e}"


def buscar_checkpoint_eq_gui(cfg_temp):
    """Pre-chequeo de checkpoint de equilibrio compatible (informativo).

    Arma las claves que la GUI conoce SIN cargar la malla y delega la
    comparación por subconjunto a constantes.buscar_checkpoint_compatible.
    Detalles que rompían la versión anterior (¡no repetir!):
      - rho_target va en C/m³ = n_e·E_CHARGE·1e27 (el motor guarda eso,
        NO el n_e en e/nm³),
      - la clave del flag de coronas se llama "rings" (como el motor),
      - el motor guarda además n_nodos/n_celdas/input_dir → por eso la
        comparación es por subconjunto y no por igualdad total del dict.
    La decisión final de reusar la toma el motor con la clave completa.
    """
    sal = cfg_temp["salt"]
    z_p = sal.get("cation", {}).get("z", 1.0)
    z_m = sal.get("anion", {}).get("z", -1.0)

    films_activos = [{
        "name": f["name"],
        "type": str(f["type"]),
        "rho_target": float(f["n_e_per_nm3"]) * E_CHARGE * 1e27,   # C/m³
    } for f in cfg_temp["films"]]

    parcial = {
        "m_name":    str(cfg_temp["m_name"]),
        "input_dir": str(cfg_temp["input_dir"]),
        "c0_mM":     float(cfg_temp["c0_mM"]),
        "sigma_Cm2": float(cfg_temp["sigma_Cm2"]),
        "rings":   bool(cfg_temp["apply_charge_rings"]),
        "T":         float(cfg_temp["T"]),
        "eps_r":     float(cfg_temp["eps_r"]),
        "z_p":       float(z_p),
        "z_m":       float(z_m),
        "films":     films_activos,
    }
    return buscar_checkpoint_compatible(parcial)



# ---------------------------------------------------------------------
# Control de sliders + number_input sincronizados
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
# Sidebar: Entradas Físicas
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("1. Geometry")
    opciones_mallas = [m["name"] for m in mallas_disponibles] + ["Enter path manually..."]
    malla_seleccionada = st.selectbox("Choose the mesh", opciones_mallas)
    
    malla_valida = False
    meta_geo = None
    malla_info = {}
    
    if malla_seleccionada == "Enter path manually...":
        col_btn, col_txt = st.columns([1, 2])
        with col_btn:
            st.write("")  # alineación
            st.write("")
            btn_buscar = st.button("📁 Browse...", use_container_width=True)
            if btn_buscar:
                ruta_sel = seleccionar_malla_local()
                if ruta_sel:
                    if ruta_sel.startswith("ERROR"):
                        st.error(f"Could not open: {ruta_sel[7:]}")
                    else:
                        st.session_state["ruta_manual_seleccionada"] = ruta_sel
                        st.rerun()

        with col_txt:
            ruta_manual = st.text_input("Mesh path or _limits.json file",
                                       value=st.session_state["ruta_manual_seleccionada"])
            if ruta_manual != st.session_state["ruta_manual_seleccionada"]:
                st.session_state["ruta_manual_seleccionada"] = ruta_manual

        ruta = st.session_state["ruta_manual_seleccionada"]
        if ruta:
            ruta = os.path.expanduser(ruta.strip())
            if not os.path.isabs(ruta):
                ruta = os.path.abspath(ruta)
            
            if os.path.isdir(ruta):
                limites = [f for f in sorted(os.listdir(ruta)) if f.endswith("_limits.json")]
                if len(limites) == 1:
                    ruta = os.path.join(ruta, limites[0])
                elif len(limites) > 1:
                    st.error(f"Found {len(limites)} *_limits.json files. Point to the file directly.")
                else:
                    st.error("No *_limits.json file found in the folder.")
            
            if os.path.isfile(ruta) and (ruta.endswith("_limits.json") or ruta.endswith("_domain.xdmf") or ruta.endswith("_facets.xdmf")):
                from constantes import derivar_malla
                input_dir, m_name = derivar_malla(ruta)
                ruta_limites = os.path.join(input_dir, f"{m_name}_limits.json")
                if not os.path.exists(ruta_limites):
                    st.error(f"Limits file not found: {ruta_limites}")
                else:
                    faltan = [f"{m_name}{suf}" for suf in ("_limits.json", "_domain.xdmf", "_facets.xdmf")
                              if not os.path.exists(os.path.join(input_dir, f"{m_name}{suf}"))]
                    if faltan:
                        st.error(f"Missing mesh files in '{input_dir}': {faltan}")
                    else:
                        malla_info = {
                            "name": m_name,
                            "ruta_json": ruta_limites,
                            "input_dir": input_dir
                        }
                        malla_valida = True
            else:
                st.error("Invalid path. Make sure to select a valid _domain.xdmf file.")

    else:
        malla_info = next(m for m in mallas_disponibles if m["name"] == malla_seleccionada)
        malla_valida = True
        
    if malla_valida:
        with open(malla_info["ruta_json"], "r") as f:
            meta_geo = json.load(f)
        # Mostrar datos descriptivos de la malla
        st.text(f"Pore: L={meta_geo['L_pore']*1e9:.1f}nm | D_tip={meta_geo.get('R_tip',0)*2e9:.1f}nm | D_base={meta_geo.get('R_base',0)*2e9:.1f}nm")

        st.divider()

        st.header("2. Electrolyte")
        catalogo_sales = cargar_catalogo_sales()
        opciones_sales = list(catalogo_sales.keys()) + ["Customize..."]
        sal_elegida = st.selectbox("Catalog salt", opciones_sales, index=opciones_sales.index(SAL_DEFAULT) if SAL_DEFAULT in opciones_sales else 0)

        sal_dict = {}
        if sal_elegida == "Customize...":
            st.caption("Custom electrolyte setup:")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Cation**")
                cat_sym = st.text_input("Symbol", "K⁺", key="cat_sym")
                cat_z = st.number_input("Valence z", value=1, step=1, key="cat_z")
                cat_D = st.number_input("D [m²/s]", value=1.96e-9, format="%.2e", key="cat_D")
            with c2:
                st.markdown("**Anion**")
                ani_sym = st.text_input("Symbol", "Cl⁻", key="ani_sym")
                ani_z = st.number_input("Valence z", value=-1, step=1, key="ani_z")
                ani_D = st.number_input("D [m²/s]", value=2.03e-9, format="%.2e", key="ani_D")

            soluble = st.checkbox("Fully soluble", value=True, key="sal_soluble")
            kps_val = None
            if not soluble:
                kps_val = st.number_input("Ksp [M²]", value=1.07e-2, format="%.2e", key="sal_kps")
                
            sal_dict = {
                "name": "custom",
                "cation": {"symbol": cat_sym, "z": cat_z, "D_m2s": cat_D},
                "anion": {"symbol": ani_sym, "z": ani_z, "D_m2s": ani_D},
                "soluble": soluble,
                "Ksp_M2": kps_val
            }
        else:
            sal_dict = catalogo_sales[sal_elegida]
            
        # c0 hasta 5000 mM: el screening del paper llegó a 3 M (3000 mM).
        c0_mM = control_valor("Bulk concentration c0 (mM)", "c0", 1.0, 5000.0, 10.0, C0_DEFAULT_MM)
        st.caption("Symmetric on both sides. Concentration gradient "
                   "(c_inlet ≠ c_outlet): **not implemented in v1** — same "
                   "limitation as the console/batch solver (see README).")
        # T y eps_r viven en "6. Avanzado" (casi nunca se tocan)

        st.divider()

        st.header("3. Surface Charge (σ)")
        sigma_e_nm2 = control_valor("Wall charge (e/nm²)", "sigma", -5.0, 5.0, 0.1, SIGMA_DEFAULT_E_NM2)
        usar_coronas = st.checkbox("Apply σ on the rings too", value=True, key="sigma_coronas")

        st.divider()

        st.header("4. Charged Films")
        films_list = []

        # Comprobar si la malla tiene films
        has_film_tip = meta_geo.get("include_film_tip", False)
        has_film_base = meta_geo.get("include_film_base", False)

        if not has_film_tip and not has_film_base:
            st.info("This mesh has no geometrically defined films.")
        else:
            for lado, activo in [("tip", has_film_tip), ("base", has_film_base)]:
                if activo:
                    st.markdown(f"**{lado.upper()} side film**")
                    # Opciones desde el catálogo (film_tipos, fuente única) — no
                    # hardcodear la lista: si se agrega un tipo, aparece solo.
                    molar_opcs = list(film_tipos.keys()) + ["Customize Molar..."]
                    idx_default = molar_opcs.index("4M") if "4M" in molar_opcs else 0
                    molar_sel = st.selectbox(f"Fixed charge (Molar) - {lado}", molar_opcs,
                                             index=idx_default, key=f"mol_sel_{lado}")

                    if molar_sel == "Customize Molar...":
                        molar_val = st.number_input("Charge [M]", value=3.0, min_value=0.01, step=0.5, key=f"mol_val_{lado}")
                    else:
                        molar_val = float(molar_sel[:-1])

                    # Positiva primero: es el default del solver de consola/batch
                    # (los tipos del catálogo, ej. "4M", son carga POSITIVA).
                    signo_sel = st.selectbox(f"Sign - {lado}",
                                             ["Positive (Anionic)", "Negative (Cationic)"],
                                             key=f"sig_sel_{lado}")
                    signo_val = 1 if "Positive" in signo_sel else -1

                    # Factor único E_NM3_TO_MOLAR (de constantes; NO hardcodear
                    # 1.6606: bug histórico documentado en constantes.py §3).
                    n_e = (molar_val / E_NM3_TO_MOLAR) * signo_val

                    # La etiqueta 'type' refleja el signo (va al _sim.json y a la
                    # clave del checkpoint): "4M" positivo, "-4M" negativo.
                    tipo_base = molar_sel if molar_sel != "Customize Molar..." \
                                else f"custom_{molar_val:.2f}M"
                    tipo = tipo_base if signo_val > 0 else f"-{tipo_base}"

                    films_list.append({
                        "name": lado,
                        "side": lado,
                        "type": tipo,
                        "molar": molar_val,
                        "signo": signo_val,
                        "n_e_per_nm3": n_e
                    })

            # Orden de rampa (el motor carga los films secuencialmente, en el
            # orden de esta lista) — misma pregunta que el solver de consola.
            if len(films_list) == 2:
                primero = st.radio("Ramp order: which film is charged first?",
                                   [f["side"] for f in films_list], horizontal=True,
                                   key="orden_rampa")
                films_list.sort(key=lambda f: 0 if f["side"] == primero else 1)

        st.divider()

        st.header("5. Voltage Sweep")
        # V_max hasta 10 V (lo que el extractor sabe escanear). Los puntos se
        # muestran SIN contar el 0 V (como los piensa el usuario: default 10 =
        # pasos de 0.1 V con V_max=1); internamente n_steps incluye el 0 V
        # (default 11), la misma convención que consola/batch.
        V_max_V = control_valor("Maximum voltage V_max (V)", "v_max", 0.05, 10.0, 0.05, V_MAX_DEFAULT_V)
        n_ptos_rama = control_valor("Points per branch (excluding 0 V)", "n_steps",
                                    1.0, 200.0, 1.0, N_STEPS_DEFAULT - 1)
        n_steps = int(n_ptos_rama) + 1   # convención interna: incluye el 0 V
        st.caption("The recipe for charged film + dilute electrolyte is to RAISE "
                   "these points (slower ramp).")

        guardar_todas = st.checkbox(
            "Save ALL U solutions in the .h5", value=False, key="guardar_todas",
            help="Unchecked: the .h5 stores the full solution U(φ, c₊, c₋) only "
                 "at 0 V and at multiples of 0.1 V (0.1, 0.2, ...) — the I-V curve "
                 "still carries ALL voltages. Checked: stores U at every point "
                 "of the sweep (much heavier file).")

        reusar_eq_check = st.checkbox(
            "Reuse saved equilibrium (checkpoint) if a compatible one exists",
            value=True, key="reusar_eq_chk",
            help="The 0 V equilibrium (σ and film ramps, ~85 MUMPS solves) is "
                 "deterministic: if it was already computed ONCE with exactly "
                 "these parameters and this mesh, it was saved in "
                 "RESULTS/equilibria/ and can be loaded in seconds instead of "
                 "recomputing it. Uncheck only to force a recompute from scratch "
                 "(e.g. if you suspect a corrupt checkpoint).")

        # ------------------------------------------------------------------
        # 6. AVANZADO — casi nunca se toca. Cada bloque muestra sus valores
        #    actuales en el título (gris/plegado) y se edita al expandirlo.
        # ------------------------------------------------------------------
        st.divider()
        st.header("6. Advanced")
        st.caption("Values that are almost never touched. The defaults are the "
                   "paper's solver ones.")

        # -- Medio: T y εr --
        _T_prev   = float(st.session_state.get("adv_T", T_DEFAULT_K))
        _eps_prev = float(st.session_state.get("adv_eps", EPS_R_DEFAULT))
        with st.expander(f"Medium: T = {_T_prev:g} K · εr = {_eps_prev:g} — ✏️ edit"):
            st.number_input("Temperature T (K)", value=T_DEFAULT_K,
                            min_value=200.0, max_value=400.0, step=0.5, key="adv_T")
            st.number_input("Dielectric constant εr", value=EPS_R_DEFAULT,
                            min_value=10.0, max_value=150.0, step=1.0, key="adv_eps")
        T_K   = float(st.session_state["adv_T"])
        eps_r = float(st.session_state["adv_eps"])

        # -- Numérica de la rampa de σ (Etapa 1) --
        _nss_prev = int(st.session_state.get("adv_nss", 10))
        _tsa_prev = float(st.session_state.get("adv_tsa", 1e-20))
        _tsr_prev = float(st.session_state.get("adv_tsr", 1e-8))
        with st.expander(f"σ ramp: {_nss_prev} steps · tol {_tsa_prev:g} / {_tsr_prev:g} — ✏️ edit"):
            st.number_input("σ ramp steps", value=10,
                            min_value=2, max_value=200, step=1, key="adv_nss")
            st.number_input("Absolute tolerance (Newton)", value=1e-20,
                            format="%e", key="adv_tsa")
            st.number_input("Relative tolerance (Newton)", value=1e-8,
                            format="%e", key="adv_tsr")
        n_steps_sigma = int(st.session_state["adv_nss"])
        tol_sigma_abs = float(st.session_state["adv_tsa"])
        tol_sigma_rel = float(st.session_state["adv_tsr"])

        # -- Numérica de la rampa de ρ_film (Etapa 1b) — solo si hay films --
        if has_film_tip or has_film_base:
            _nsf_prev = int(st.session_state.get("adv_nsf", N_STEPS_FILM))
            _tfa_prev = float(st.session_state.get("adv_tfa", 1e-20))
            _tfr_prev = float(st.session_state.get("adv_tfr", 1e-8))
            with st.expander(f"ρ_film ramp: {_nsf_prev} steps · tol {_tfa_prev:g} / {_tfr_prev:g} — ✏️ edit"):
                st.number_input("ρ_film ramp steps", value=int(N_STEPS_FILM),
                                min_value=5, max_value=400, step=5, key="adv_nsf")
                st.number_input("Absolute tolerance (Newton)", value=1e-20,
                                format="%e", key="adv_tfa")
                st.number_input("Relative tolerance (Newton)", value=1e-8,
                                format="%e", key="adv_tfr")
            n_steps_film = int(st.session_state["adv_nsf"])
            tol_film_abs = float(st.session_state["adv_tfa"])
            tol_film_rel = float(st.session_state["adv_tfr"])
        else:
            n_steps_film = int(N_STEPS_FILM)
            tol_film_abs, tol_film_rel = 1e-20, 1e-8

# Control de parada del layout principal si no hay malla válida cargada
if not malla_valida:
    if not mallas_disponibles:
        st.warning(f"⚠️ No pre-scanned meshes found in the repository folder:\n`{RESULTADOS_MALLAS}`\n\n**Please enter or browse a manual path in the left panel to begin.**")
    else:
        st.info("👋 Please select a mesh in the left panel to start configuring the simulation.")
    st.stop()


# ---------------------------------------------------------------------
# Preparar estados geométricos para el graficador físico
# ---------------------------------------------------------------------
# Crear diccionario de estado geométrico compatible con dibujador
st_estado = {
    "type": meta_geo.get("channel_type", "conical"),
    "D_tip": meta_geo.get("R_tip", 5e-9)*2,
    "D_base": meta_geo.get("R_base", 25e-9)*2,
    "L_pore": meta_geo["L_pore"],
    "L_charge": meta_geo.get("L_charge", 0.0),
    "L_far": meta_geo.get("L_far", 0.0),
    "usar_corona": meta_geo.get("usar_corona", False),
    "L_res": meta_geo.get("L_res"),
    "R_res": meta_geo.get("R_res"),
    "h_param": meta_geo.get("h_param"),
    "film_tip": {"delta": meta_geo["delta_film_tip"]} if has_film_tip else None,
    "film_base": {"delta": meta_geo["delta_film_base"]} if has_film_base else None
}


# ---------------------------------------------------------------------
# Layout Principal (Dos columnas: Visualización y Proto-Plot)
# ---------------------------------------------------------------------
placeholder_warnings = st.empty()

col_vis, col_proto = st.columns(2)

with col_vis:
    st.subheader("Physical Setup")
    fig_v, ax_v = plt.subplots(figsize=(6, 5))
    dibujar_canal_fisica(ax_v, st_estado, sigma_e_nm2, usar_coronas, films_list)
    ax_v.legend(handles=leyenda_handles_fisica(sigma_e_nm2, films_list), 
                loc="upper right", fontsize=6, framealpha=0.9)
    st.pyplot(fig_v)
    plt.close(fig_v)

with col_proto:
    st.subheader("Sweep Sampling")
    proto_placeholder = st.empty()
    fig_p, ax_p = plt.subplots(figsize=(6, 5))
    dibujar_proto_plot(ax_p, V_max_V, int(n_steps))
    proto_placeholder.pyplot(fig_p)
    plt.close(fig_p)


# ---------------------------------------------------------------------
# Parámetros de Salida e Inicio de Simulación
# ---------------------------------------------------------------------
st.divider()
st.subheader("Run Simulation")

# Evaluar si hay checkpoint compatible en tiempo real
sigma_Cm2_eval = sigma_e_nm2 * E_CHARGE / (1e-9)**2
cfg_temp = {
    "m_name": malla_info["name"],
    "input_dir": malla_info["input_dir"],
    "T": T_K,
    "eps_r": eps_r,
    "salt": sal_dict,
    "c0_mM": c0_mM,
    "sigma_Cm2": sigma_Cm2_eval,
    "apply_charge_rings": usar_coronas,
    "films": [{"name": f["name"], "type": f["type"], "n_e_per_nm3": f["n_e_per_nm3"]} for f in films_list]
}
chk_h5 = buscar_checkpoint_eq_gui(cfg_temp)

if chk_h5:
    st.success(f"✓ **Compatible equilibrium found on disk:** `{os.path.basename(chk_h5)}`. It will be reused to skip the initial equilibrium computation (Stages 1/1b), speeding up the simulation.")
else:
    st.info("ℹ️ **No compatible checkpoint:** the equilibrium state (V=0) will be solved from scratch before starting the voltage ramps.")

col_out, col_run = st.columns([2, 1])

with col_out:
    # Mismo formato que el default del batch/consola (c0 como float → "100.0mM"),
    # así GUI y batch escriben en la MISMA subcarpeta para la misma corrida.
    nombre_defecto = f"{malla_info['name']}_{sal_dict['name']}_{c0_mM}mM"
    salida_sub = st.text_input("Output subfolder (inside RESULTS/solutions/)", value=nombre_defecto)
    output_dir = os.path.join(RESULTADOS_SOLUCIONES, salida_sub)
    st.caption(f"Outputs saved in: `{output_dir}`")

with col_run:
    st.write("") # espacio
    st.write("")
    btn_simular = st.button("🚀 Launch PNP Simulation", type="primary", use_container_width=True)

# Contenedor para mostrar los resultados de la simulación
res_container = st.container()

if btn_simular:
    # 1. Armar el diccionario cfg
    sigma_Cm2 = sigma_e_nm2 * E_CHARGE / (1e-9)**2
    
    cfg = {
        "input_dir": malla_info["input_dir"],
        "m_name": malla_info["name"],
        "output_dir": output_dir,
        "T": T_K,
        "eps_r": eps_r,
        "salt": sal_dict,
        "c0_mM": c0_mM,
        "sigma_Cm2": sigma_Cm2,
        "apply_charge_rings": usar_coronas,
        "films": [{"name": f["name"], "type": f["type"], "n_e_per_nm3": f["n_e_per_nm3"]} for f in films_list],
        "V_max_V": V_max_V,
        "n_steps": int(n_steps),
        "guardar_todas_sol": guardar_todas,
        "n_steps_film": int(n_steps_film),
        "reuse_equilibrium": "auto" if reusar_eq_check else False,
        # perillas numéricas avanzadas (defaults = paper si no se tocaron)
        "n_steps_sigma": n_steps_sigma,
        "tol_sigma_abs": tol_sigma_abs,
        "tol_sigma_rel": tol_sigma_rel,
        "tol_film_abs":  tol_film_abs,
        "tol_film_rel":  tol_film_rel,
    }

    # Guardar la última malla como activa
    try:
        from constantes import HISTORIAL_MALLA
        os.makedirs(os.path.dirname(HISTORIAL_MALLA), exist_ok=True)
        with open(HISTORIAL_MALLA, "w") as f_hist:
            f_hist.write(os.path.join(malla_info["input_dir"], f"{malla_info['name']}_limits.json"))
    except Exception:
        pass

    # 2. Correr el motor PNP
    with res_container:
        def _graficar_iv_en_vivo(puntos_iv):
            if not puntos_iv:
                return
            arr = np.array(puntos_iv)
            arr = arr[arr[:, 0].argsort()]
            fig_live, ax_live = plt.subplots(figsize=(6, 5))
            ax_live.plot(arr[:, 0], -arr[:, 1], "o-", color="#2980b9", label="-I_in")
            ax_live.plot(arr[:, 0], arr[:, 2], "s--", color="#e74c3c", label="I_out")
            ax_live.axhline(0, color="gray", lw=0.8, ls=":")
            ax_live.axvline(0, color="gray", lw=0.8, ls=":")
            ax_live.set_xlabel("Applied voltage V [V]")
            ax_live.set_ylabel("Current I [nA]")
            ax_live.set_title(f"⚡ Live PNP sweep ({len(puntos_iv)} points)")
            ax_live.grid(True, ls=":", alpha=0.4)
            ax_live.legend(loc="best", fontsize=8)
            proto_placeholder.pyplot(fig_live)
            plt.close(fig_live)

        cfg["callback_iv"] = _graficar_iv_en_vivo

        with st.spinner("Running the FEniCS PNP solver (this may take a few minutes)..."):
            # Capturar logs internos
            from motor_pnp import resolver
            try:
                # Ejecutar el solver
                res = resolver(cfg)
                
                st.success("Simulation completed successfully!")

                # Mostrar los paths generados
                st.info(f"📂 **HDF5 (Mesh + Solutions):** `{res['archivo_h5']}`\n\n"
                        f"📂 **JSON metadata:** `{res['archivo_json']}`\n\n"
                        f"📂 **I-V current table:** `{res['archivo_txt']}`")
                
                # Avisos de calidad del solver (fuente única: res["warnings"]).
                # Se renderizan por severidad; tolera tanto dicts como strings.
                warns = res.get("warnings", [])
                if warns:
                    with placeholder_warnings.container():
                        st.subheader("⚠️ Quality warnings")
                        
                        # Agrupar por código para evitar un cartel gigante si fallan muchos voltajes
                        from collections import defaultdict
                        grupos = defaultdict(list)
                        for w in warns:
                            if isinstance(w, dict) and "code" in w:
                                grupos[w["code"]].append(w)
                            else:
                                grupos["otros"].append(w)
                                
                        for code, items in grupos.items():
                            if len(items) == 1 or code == "otros":
                                for w in items:
                                    texto = w["msg"] if isinstance(w, dict) else str(w)
                                    nivel = w.get("level", "warning") if isinstance(w, dict) else "warning"
                                    (st.info if nivel == "note" else st.warning)(texto)
                            else:
                                nivel = items[0].get("level", "warning")
                                voltajes = [f"{w.get('v', 0):+.2f}V" for w in items if "v" in w]
                                texto = f"**{len(items)} advertencias de tipo '{code}'** en los voltajes: {', '.join(voltajes)}.\n\n*Ejemplo del mensaje:* {items[0]['msg']}"
                                (st.info if nivel == "note" else st.warning)(texto)

                # 3. Graficar la curva I-V resuelta inmediatamente
                iv_data = np.array(res["IV"]) # Columnas: (V, I_in, I_out)
                # Ordenar por voltaje para graficar correctamente
                iv_data = iv_data[iv_data[:, 0].argsort()]
                
                st.subheader("Computed I-V curve")
                fig_res, ax_res = plt.subplots(figsize=(10, 5))

                # Convención de signos:
                # I_in se exporta como -I_in (entrante positiva)
                # I_out como +I_out
                ax_res.plot(iv_data[:, 0], -iv_data[:, 1], "o-", color="#2980b9", label="Inlet current (-I_in)")
                ax_res.plot(iv_data[:, 0], iv_data[:, 2], "s--", color="#e74c3c", label="Outlet current (I_out)")

                ax_res.axhline(0, color="gray", lw=0.8, ls=":")
                ax_res.axvline(0, color="gray", lw=0.8, ls=":")
                ax_res.set_xlabel("Applied voltage V [V]")
                ax_res.set_ylabel("Current I [nA]")
                ax_res.set_title("Solver results: I-V curve")
                ax_res.grid(True, ls=":", alpha=0.3)
                ax_res.legend()

                st.pyplot(fig_res)
                plt.close(fig_res)

                # Tabla de corriente. El error de conservación usa la MISMA
                # métrica que el motor (|I_in+I_out| / magnitud media) para que
                # el % de la tabla coincida con el que dispara los avisos.
                mag_media = (np.abs(-iv_data[:, 1]) + np.abs(iv_data[:, 2])) / 2.0
                st.dataframe({
                    "Voltage (V)": iv_data[:, 0],
                    "I_in (nA)": -iv_data[:, 1],
                    "I_out (nA)": iv_data[:, 2],
                    "Conservation error (%)": np.abs(-iv_data[:, 1] - iv_data[:, 2]) / (mag_media + 1e-12) * 100
                })

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
