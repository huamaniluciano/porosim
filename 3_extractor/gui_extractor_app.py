# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — EXTRACTOR · GUI Streamlit (versión MODULAR)
  Se corre con:   python launch_extractor.py
  (o directo: streamlit run gui_extractor_app.py)
═══════════════════════════════════════════════════════════════════════════

Dashboard científico sobre una SOLUCIÓN del solver (Pilar 2):
  · Pestaña A — Barrido completo: curva I-V, factor de rectificación a ±1 V
    y conductancia diferencial dI/dV (de la tabla IV_curve_*.txt).
  · Pestaña B — Física interna a un voltaje: perfiles axiales (r=0) de φ y
    concentraciones, mapas 2D de potencial (± líneas de campo), iones
    totales y precipitación de Davies (solo sales poco solubles).

DIFERENCIA CLAVE vs. la versión original: esta GUI NO reimplementa la física
ni el graficado. Importa los MÓDULOS de modulos/<categoria>/ (los mismos que
usa el modo consola de extractor.py) y llama a sus funciones crear_figura() /
primitivas de trazado. Editar un módulo repercute acá y en la consola a la vez.
La única lógica propia de la GUI es la CAPA DE CACHÉ (envuelve la carga FEniCS
de porosim_comun con st.cache_*) y el panel de exportación de cada gráfico.
"""
import os
import sys
import pathlib

# ─── porosim_comun prepara el entorno FEniCS (MPI/UCX/threads) ──────────────
AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "modulos"))
for _cat in ("potential_map", "ion_maps", "precipitation_map", "axial_profiles"):
    sys.path.insert(0, str(AQUI / "modulos" / _cat))

import importlib
import porosim_comun as pc
pc.preparar_entorno_fenics()

import io
import json

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Módulos del extractor (única fuente de verdad de física + figuras) ─────
import precipitation
import potential
import field_lines          # noqa: F401 (mismo crear_figura que potencial con campo)
import ions
import total_ions
import axis_profile_potential as perf_pot
import axis_profile_ions as perf_ion
perf_prom = importlib.import_module("section_avg_concentration")

try:
    import pandas as pd
except ImportError:
    pd = None

_REPO_ROOT            = AQUI.parent
RESULTADOS_SOLUCIONES = _REPO_ROOT / "RESULTS" / "solutions"

E_CHARGE = pc.E_CHARGE
MIMES = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}

# Mapeo etiqueta GUI → clave de especie de los módulos
_ESPECIE = {"Cation": "cation", "Anion": "anion", "Total ions": "total"}


def _curvas(seleccion):
    """['Cation','Total ions'] → ('cation','total') para las primitivas."""
    return tuple(_ESPECIE[s] for s in seleccion if s in _ESPECIE)


st.set_page_config(page_title="PoroSim — Extractor", layout="wide")
st.title("PoroSim · Extractor")
st.caption("I-V curve of the full sweep and internal physics of the nanopore "
           "(axial profiles and 2D maps) at each computed voltage.")


# =============================================================================
# CAPA DE CACHÉ (envuelve porosim_comun con st.cache_*)
# La malla FEniCS se carga 1 vez por .h5 (recurso NO serializable); los campos
# nodales se extraen a arrays NumPy serializables. mtime entra en la clave para
# invalidar si el archivo cambia en disco.
# =============================================================================
@st.cache_resource(max_entries=3, show_spinner="Loading FEniCS mesh from the .h5 ...")
def cargar_malla(h5_path, mtime):
    mesh = pc.cargar_malla(h5_path)
    return {"mesh": mesh, "V": pc.espacio_mixto(mesh)}


@st.cache_data(max_entries=6)
def datos_malla(h5_path, mtime):
    return pc.datos_malla(cargar_malla(h5_path, mtime)["mesh"])


@st.cache_data(max_entries=6)
def detectar_voltajes(h5_path, mtime, V_max, n_steps):
    mesh = cargar_malla(h5_path, mtime)["mesh"]
    return pc.detectar_voltajes(h5_path, mesh,
                                {"V_max_V": V_max, "n_steps": n_steps})


@st.cache_data(max_entries=24, show_spinner="Extracting nodal fields from the .h5 ...")
def extraer_campos(h5_path, mtime, v_label):
    rec = cargar_malla(h5_path, mtime)
    return pc.leer_campos(h5_path, rec["mesh"], rec["V"], v_label)


@st.cache_data(max_entries=8, show_spinner="Computing E = -∇φ on the grid ...")
def campo_electrico(h5_path, mtime, v_label, V_T):
    dm    = datos_malla(h5_path, mtime)
    phi_V = extraer_campos(h5_path, mtime, v_label)["phi_adim"] * V_T
    return potential.campo_electrico(dm["z_nm"], dm["r_nm"], dm["tri"], phi_V)


@st.cache_data(max_entries=12, show_spinner="Computing <c>(z) on the cross-section ...")
def perfil_concentracion_promedio(z_nm, r_nm, tri, cp, cm,
                                   z_tip, z_base, R_tip, R_base, c0):
    return perf_prom.perfil_promedio_seccion(z_nm, r_nm, tri, cp, cm,
                                             z_tip, z_base, R_tip, R_base, c0)


def campos_mM(h5_path, mtime, v_label, c0):
    """cp, cm en mM (c = c0·exp(u)) desde los campos cacheados."""
    campos = extraer_campos(h5_path, mtime, v_label)
    return c0 * np.exp(campos["up"]), c0 * np.exp(campos["um"])


def datos_2d(h5_path, mtime, v_label, c0):
    """dict {z_nm, r_nm, tri, cp_mM, cm_mM} que consumen las crear_figura 2D."""
    dm = datos_malla(h5_path, mtime)
    cp, cm = campos_mM(h5_path, mtime, v_label, c0)
    return {"z_nm": dm["z_nm"], "r_nm": dm["r_nm"], "tri": dm["tri"],
            "cp_mM": cp, "cm_mM": cm}


# =============================================================================
# HERRAMIENTAS COMUNES DE GRAFICADO (panel de exportación — propio de la GUI)
# =============================================================================
def panel_grafico(key, dibujar, carpeta_sol, nombre_base, datos=None):
    """Renderiza un gráfico + su panel expandible de herramientas.

    dibujar(mostrar_leyenda, limpio) -> fig. Las opciones se leen de
    session_state ANTES de dibujar (los widgets de abajo las escriben y el
    re-run de Streamlit regenera la figura), así el panel queda DEBAJO del
    gráfico sin quedar un paso atrás.
    datos: dict opcional {"df": DataFrame, "header": str} para exportar tabla.
    """
    leyenda = bool(st.session_state.get(f"{key}__ley", True))
    limpio  = bool(st.session_state.get(f"{key}__limpio", False))

    fig = dibujar(leyenda, limpio)
    if limpio:
        pc.limpiar_figura(fig)
    st.pyplot(fig)

    with st.expander("🛠 Style / saving / export"):
        c1, c2, c3, c4 = st.columns(4)
        c1.checkbox("Show legend", value=True, key=f"{key}__ley")
        c2.checkbox("Clean image (no axes or title)", value=False, key=f"{key}__limpio")
        dpi = c3.selectbox("DPI", [150, 300, 600], index=1, key=f"{key}__dpi")
        fmt = c4.selectbox("Format", ["png", "pdf", "svg"], key=f"{key}__fmt")

        buf = io.BytesIO()
        try:
            fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        except Exception as e:
            st.error(f"Could not render the image: {e}")
            buf = None

        if buf is not None:
            nombre_img = f"{nombre_base}.{fmt}"
            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Save to the solution folder",
                             key=f"{key}__save", use_container_width=True):
                    destino = carpeta_sol / nombre_img
                    with open(destino, "wb") as f:
                        f.write(buf.getvalue())
                    st.success(f"Saved: `{destino}`")
            with b2:
                st.download_button("⬇️ Download to browser", data=buf.getvalue(),
                                   file_name=nombre_img, mime=MIMES[fmt],
                                   key=f"{key}__dl", use_container_width=True)

        if datos is not None and pd is not None:
            df  = datos["df"]
            enc = datos.get("header", "")
            txt = (f"# {enc}\n" if enc else "") + df.to_csv(sep="\t", index=False,
                                                            float_format="%.6e")
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Data .txt", txt, file_name=f"{nombre_base}.txt",
                               mime="text/plain", key=f"{key}__dltxt",
                               use_container_width=True)
            try:
                xbuf = io.BytesIO()
                df.to_excel(xbuf, index=False)
                d2.download_button("⬇️ Data .xlsx", xbuf.getvalue(),
                                   file_name=f"{nombre_base}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key=f"{key}__dlxlsx", use_container_width=True)
            except Exception:
                d2.caption("(.xlsx unavailable: openpyxl missing)")
            with d3:
                if st.button("💾 Data .txt in the folder", key=f"{key}__savetxt",
                             use_container_width=True):
                    destino = carpeta_sol / f"{nombre_base}.txt"
                    destino.write_text(txt, encoding="utf-8")
                    st.success(f"Saved: `{destino}`")

    plt.close(fig)


# =============================================================================
# SIDEBAR: selector de solución + tarjeta de metadatos
# =============================================================================
def buscar_soluciones():
    """Solutions_*.h5 dentro de RESULTS/solutions/ (recursivo)."""
    if not RESULTADOS_SOLUCIONES.is_dir():
        return []
    return sorted(RESULTADOS_SOLUCIONES.glob("**/Solutions_*.h5"))


def seleccionar_h5_local():
    """Diálogo tkinter nativo para buscar un .h5 fuera de RESULTS."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        inicio = str(RESULTADOS_SOLUCIONES) if RESULTADOS_SOLUCIONES.is_dir() \
                 else str(_REPO_ROOT)
        ruta = filedialog.askopenfilename(
            title="Choose the solution file (.h5)",
            initialdir=inicio,
            filetypes=[("FEniCS solutions", "*.h5"), ("All", "*.*")])
        root.destroy()
        return ruta
    except Exception as e:
        return f"ERROR: {e}"


if "ruta_manual_h5" not in st.session_state:
    st.session_state["ruta_manual_h5"] = ""

soluciones = buscar_soluciones()
ruta_solucion = None

with st.sidebar:
    st.header("1. Solution")
    opciones = [str(p.relative_to(RESULTADOS_SOLUCIONES)) for p in soluciones] \
               + ["Enter path manually..."]
    sel = st.selectbox("Solver run (RESULTS/solutions/)", opciones)

    if sel == "Enter path manually...":
        col_btn, col_txt = st.columns([1, 2])
        with col_btn:
            st.write("")
            st.write("")
            if st.button("📁 Browse...", use_container_width=True):
                ruta_sel = seleccionar_h5_local()
                if ruta_sel:
                    if str(ruta_sel).startswith("ERROR"):
                        st.error(f"Could not open the file browser: {ruta_sel[7:]}")
                    else:
                        st.session_state["ruta_manual_h5"] = ruta_sel
                        st.rerun()
        with col_txt:
            ruta_manual = st.text_input("Path to Solutions_*.h5 (or its folder)",
                                        value=st.session_state["ruta_manual_h5"])
            if ruta_manual != st.session_state["ruta_manual_h5"]:
                st.session_state["ruta_manual_h5"] = ruta_manual

        ruta = st.session_state["ruta_manual_h5"].strip()
        if ruta:
            p = pathlib.Path(os.path.expanduser(ruta))
            if not p.is_absolute():
                p = pathlib.Path(os.path.abspath(str(p)))
            if p.is_dir():
                h5s = sorted(p.glob("Solutions_*.h5")) or sorted(p.glob("*.h5"))
                if len(h5s) == 1:
                    p = h5s[0]
                elif len(h5s) > 1:
                    st.error(f"The folder has {len(h5s)} .h5. Point to the file directly.")
                    p = None
                else:
                    st.error("No .h5 found in that folder.")
                    p = None
            if p is not None:
                if p.is_file() and p.suffix == ".h5":
                    ruta_solucion = p
                else:
                    st.error(f"Invalid path: `{ruta}` (expected a solver .h5).")
    else:
        ruta_solucion = RESULTADOS_SOLUCIONES / sel

if ruta_solucion is None:
    if not soluciones:
        st.warning(f"⚠️ No solutions found in `{RESULTADOS_SOLUCIONES}`.\n\n"
                   "**Run the solver (Pillar 2) or enter/browse a manual path in "
                   "the left panel to begin.**")
    else:
        st.info("👋 Select a solution in the left panel to begin.")
    st.stop()

# ─── Metadatos de la corrida (lectura defensiva del _sim.json vía pc) ───────
carpeta_sol = ruta_solucion.parent
h5_path     = str(ruta_solucion)
mtime       = os.path.getmtime(h5_path)

meta = pc.cargar_meta(carpeta_sol)
sim  = meta.get("simulation", {})
if not meta:
    st.warning("⚠️ Could not find (or read) the sibling *_sim.json. Continuing with "
               "legacy defaults: KCl, c0 = 100 mM, T = 298.15 K. The absolute "
               "concentration/potential values may not match.")

stem = ruta_solucion.stem
ctx  = pc.contexto_de(meta, stem)           # contexto base (sin voltaje aún)
sal   = ctx["salt"]
c0    = ctx["c0"]
T_K   = ctx["T_K"]
V_T   = ctx["V_T"]
films = ctx["films"]

with st.sidebar:
    st.header("2. Run metadata")
    sigma_Cm2 = ctx["sigma_Cm2"]
    lineas = [f"**Mesh**: `{meta.get('m_name', stem)}` "
              f"({meta.get('channel_type', '?')})"]
    if meta.get("R_tip") is not None:
        lineas.append(f"- D_tip = {meta['R_tip']*2e9:.1f} nm · "
                      f"D_base = {meta.get('R_base', 0)*2e9:.1f} nm · "
                      f"L_pore = {meta.get('L_pore', 0)*1e9:.0f} nm")
    lineas.append(f"**Electrolyte**: {sal['name']} · {c0:g} mM"
                  + (" *(legacy)*" if sal["legacy"] else ""))
    lineas.append(f"- {sal['label_p']} / {sal['label_m']} · T = {T_K:g} K · "
                  f"ε_r = {sim.get('eps_r', '?')}")
    if sal["soluble"] is False and sal["Ksp_M2"] is not None:
        lineas.append(f"- Sparingly soluble · Ksp = {sal['Ksp_M2']:.2e} M²")
    if sigma_Cm2 is not None:
        sigma_e = sigma_Cm2 * 1e-18 / E_CHARGE
        lineas.append(f"**Wall charge**: σ = {sigma_e:+.2f} e/nm² "
                      f"({sigma_Cm2:+.3f} C/m²)")
    if films:
        for f_ in films:
            signo = "+" if f_["rho"] > 0 else "−"
            lineas.append(f"**Film {f_['side']}**: {f_['type']} (charge {signo})")
    else:
        lineas.append("**Films**: no films")
    if sim.get("V_max_V") is not None:
        lineas.append(f"**Sweep**: ±{sim['V_max_V']:g} V · "
                      f"{sim.get('n_steps', '?')} points per branch")
    st.markdown("\n".join(lineas))
    st.caption(f"Folder: `{carpeta_sol}`")

st.subheader(f"Solution: `{ruta_solucion.name}`")

tab_global, tab_local = st.tabs(["📈 Full-sweep analysis",
                                 "🔬 Internal physics (per voltage)"])


def ctx_v(v_label):
    """Contexto del módulo (pc.contexto_de) para un voltaje concreto."""
    return pc.contexto_de(meta, stem, v_label)


# =============================================================================
# PESTAÑA A — BARRIDO COMPLETO: I-V, rectificación, dI/dV
# (no usa los módulos: lee IV_curve_*.txt, contrato directo del solver)
# =============================================================================
def cargar_curva_iv(carpeta):
    candidatos = sorted(carpeta.glob("curva_IV*.txt")) or sorted(carpeta.glob("*IV*.txt"))
    if not candidatos:
        return None
    try:
        datos = np.atleast_2d(np.loadtxt(candidatos[0], skiprows=1))
        return datos[datos[:, 0].argsort()]
    except Exception:
        return None


with tab_global:
    iv = cargar_curva_iv(carpeta_sol)
    if iv is None or iv.shape[0] < 2 or iv.shape[1] < 3:
        st.warning("Could not find a readable `IV_curve_*.txt` table in the "
                   "solution folder (the solver generates it next to the .h5). Without it "
                   "I cannot show the I-V curve or dI/dV.")
    else:
        V_arr, I_in, I_out = iv[:, 0], iv[:, 1], iv[:, 2]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Sweep points", f"{len(V_arr)}")
        col_m2.metric("Voltage range", f"{V_arr.min():+.2f} … {V_arr.max():+.2f} V")

        v_pos = V_arr[V_arr > 1e-9]
        v_neg = V_arr[V_arr < -1e-9]
        if len(v_pos) and len(v_neg):
            v1 = v_pos[np.argmin(np.abs(v_pos - 1.0))]
            v2 = v_neg[np.argmin(np.abs(v_neg + 1.0))]
            i1 = I_in[np.argmin(np.abs(V_arr - v1))]
            i2 = I_in[np.argmin(np.abs(V_arr - v2))]
            if abs(i2) > 1e-12:
                col_m3.metric(f"Rectification factor ({v1:+.2f}/{v2:+.2f} V)",
                              f"{i1 / abs(i2):.2f}",
                              help="r_f = I(+1 V) / |I(−1 V)| on the Inlet "
                                   "current, at the available voltage closest to ±1 V.")
            else:
                col_m3.metric("Rectification factor", "—")
        else:
            col_m3.metric("Rectification factor", "—",
                          help="The sweep does not have both branches (+ and −).")

        def dibujar_iv(leyenda, limpio):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(V_arr, I_in, "o-", color="#2980b9", ms=4, lw=1.5,
                    label="Inlet current")
            ax.plot(V_arr, I_out, "s--", color="#e74c3c", ms=3.5, lw=1.2,
                    label="Outlet current")
            ax.axhline(0, color="gray", lw=0.8, ls=":")
            ax.axvline(0, color="gray", lw=0.8, ls=":")
            ax.set_xlabel("Applied voltage V [V]")
            ax.set_ylabel("Current I [nA]")
            ax.set_title(f"I-V curve — {stem}")
            ax.grid(True, ls=":", alpha=0.3)
            if leyenda:
                ax.legend()
            fig.tight_layout()
            return fig

        datos_iv = None
        if pd is not None:
            datos_iv = {"df": pd.DataFrame({"Voltage_V": V_arr, "I_in_nA": I_in,
                                            "I_out_nA": I_out}),
                        "header": f"Curva I-V | {stem}"}
        panel_grafico("iv", dibujar_iv, carpeta_sol, f"{stem}_IV_curve", datos_iv)

        st.divider()

        g_in  = np.gradient(I_in, V_arr)
        g_out = np.gradient(I_out, V_arr)

        def dibujar_didv(leyenda, limpio):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(V_arr, g_in, "o-", color="#2980b9", ms=4, lw=1.5,
                    label="dI/dV Inlet")
            ax.plot(V_arr, g_out, "s--", color="#e74c3c", ms=3.5, lw=1.2,
                    label="dI/dV Outlet")
            ax.axvline(0, color="gray", lw=0.8, ls=":")
            ax.set_xlabel("Applied voltage V [V]")
            ax.set_ylabel("Differential conductance dI/dV [nS]")
            ax.set_title(f"Differential conductance — {stem}")
            ax.grid(True, ls=":", alpha=0.3)
            if leyenda:
                ax.legend()
            fig.tight_layout()
            return fig

        datos_g = None
        if pd is not None:
            datos_g = {"df": pd.DataFrame({"Voltage_V": V_arr, "dIdV_in_nS": g_in,
                                           "dIdV_out_nS": g_out}),
                       "header": f"Conductancia diferencial (np.gradient) | {stem}"}
        panel_grafico("didv", dibujar_didv, carpeta_sol, f"{stem}_dIdV", datos_g)

        st.divider()

        with st.expander("📋 Full Technical Report of the Solution (solution_summary.py)"):
            lineas_rep = [
                f"=== POROSIM — SOLUTION REPORT ===",
                f"File: {ruta_solucion.name}",
                f"Folder: {carpeta_sol}",
                f"Mesh:    {meta.get('m_name', stem)} ({meta.get('channel_type', '?')})",
                f"Salt:    {sal['name']} | c0 = {c0:g} mM | T = {T_K:g} K | ε_r = {sim.get('eps_r', '?')}",
                f"Ions:    {sal['label_p']} (z={sal['z_p']}) / {sal['label_m']} (z={sal['z_m']})"
            ]
            if meta.get("R_tip") is not None:
                lineas_rep.append(f"Geometry: D_tip = {meta['R_tip']*2e9:.1f} nm, D_base = {meta.get('R_base', 0)*2e9:.1f} nm, L = {meta.get('L_pore', 0)*1e9:.0f} nm")
            lineas_rep.append(f"Points in I-V sweep: {len(V_arr)}")
            lineas_rep.append(f"Voltage range: {V_arr.min():+.2f} V … {V_arr.max():+.2f} V")
            texto_reporte = "\n".join(lineas_rep) + "\n\n=== TABLA I-V ===\n" + (
                pd.DataFrame({"V_applied_V": V_arr, "I_in_nA": I_in, "I_out_nA": I_out}).to_string(index=False)
                if pd is not None else ""
            )
            st.text(texto_reporte)
            c_dl1, c_dl2 = st.columns(2)
            c_dl1.download_button("⬇️ Download Report (.txt)", texto_reporte,
                                  file_name=f"{stem}_summary_report.txt", mime="text/plain",
                                  use_container_width=True, key="rep__dl")
            if c_dl2.button("💾 Save report to solution folder", use_container_width=True, key="rep__save"):
                destino_rep = carpeta_sol / f"{stem}_summary_report.txt"
                destino_rep.write_text(texto_reporte, encoding="utf-8")
                st.success(f"Report saved in: {destino_rep}")


# =============================================================================
# PESTAÑA B — FÍSICA INTERNA A UN VOLTAJE (usa las crear_figura de los módulos)
# =============================================================================
with tab_local:
    try:
        labels_v = detectar_voltajes(h5_path, mtime,
                                     sim.get("V_max_V"), sim.get("n_steps"))
    except Exception as e:
        st.error(f"Could not read the .h5 (is it a solver solution?): {e}")
        labels_v = []

    if not labels_v:
        st.error("The .h5 has no solution datasets `/U_±X.XXV`. "
                 "Is it a file generated by the solver (Pillar 2)?")
    else:
        default_v = min(labels_v, key=lambda l: abs(float(l) + 1.0))

        st.markdown("### ⚡ Analysis Voltage Selection")
        v_label = st.select_slider(
            "Sweep voltage to visualize",
            options=labels_v, value=default_v,
            format_func=lambda l: f"{l} V",
            help=f"{len(labels_v)} solutions stored in the .h5. "
                 "All physical plots depend on this main voltage."
        )
        st.divider()

        sub_nombres = ["📉 1D profiles in the channel",
                       "🗺 2D potential + field", "🧂 2D concentration maps"]
        hay_precip = precipitation.aplica(meta)
        if hay_precip:
            sub_nombres.append("💎 Precipitation (Davies)")
        vista = st.radio("View", sub_nombres, horizontal=True,
                         label_visibility="collapsed", key="vista_local")

        # ── B.1: PERFILES 1D EN EL CANAL ────────────────────────────────────
        if vista == sub_nombres[0]:
            tab_axial, tab_prom = st.tabs(["📍 Central axis (r = 0)",
                                           "⭕ Cross-section average <c>(z)"])

            # ── Eje central r=0 (compone las 2 primitivas de los módulos) ──
            with tab_axial:
                with st.form("form_perf_axial", border=True):
                    modo_ax = st.radio(
                        "Visualization mode at r=0",
                        ["Concentration + Potential (2 stacked panels)",
                         "Concentration only (ions)",
                         "Potential only (φ(z))"],
                        horizontal=True
                    )
                    if "Concentration" in modo_ax:
                        curvas_ax = st.multiselect(
                            "Ion species to show",
                            ["Cation", "Anion", "Total ions"],
                            default=["Cation", "Anion", "Total ions"]
                        )
                    else:
                        curvas_ax = []
                    btn_ax = st.form_submit_button(f"🧮 Compute and Plot ({v_label} V)", type="primary", use_container_width=True)

                if btn_ax:
                    st.session_state["v_calc__perf_ax"] = v_label

                if st.session_state.get("v_calc__perf_ax") != v_label:
                    st.info(f"👆 Selected voltage: **{v_label} V**. Adjust the options and click **🧮 Compute and Plot ({v_label} V)**.")
                else:
                    ctxv   = ctx_v(v_label)
                    dm     = datos_malla(h5_path, mtime)
                    campos = extraer_campos(h5_path, mtime, v_label)
                    ai     = dm["axis_idx"]
                    phi_ax = campos["phi_adim"][ai] * V_T
                    cp_ax  = c0 * np.exp(campos["up"][ai])
                    cm_ax  = c0 * np.exp(campos["um"][ai])
                    ct_ax  = cp_ax + cm_ax
                    z_axis = dm["z_nm"][ai]

                    d_pot = {"z_axis_nm": z_axis, "phi_ax_V": phi_ax}
                    d_ion = {"z_axis_nm": z_axis, "cp_ax_mM": cp_ax,
                             "cm_ax_mM": cm_ax, "ct_ax_mM": ct_ax}
                    curvas = _curvas(curvas_ax)

                    def dibujar_perf_axial(leyenda, limpio):
                        ver_pot = ("Potential" in modo_ax)
                        ver_con = ("Concentration" in modo_ax)
                        if ver_pot and ver_con:
                            fig, (ax_f, ax_c) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
                        elif ver_pot:
                            fig, ax_f = plt.subplots(figsize=(11, 4.5)); ax_c = None
                        else:
                            fig, ax_c = plt.subplots(figsize=(11, 5)); ax_f = None
                        fig.suptitle(f"Central Axis Profile (r=0) — {v_label} V | {stem}", fontsize=12)
                        if ax_f is not None:
                            perf_pot.dibujar_potencial_axial(
                                ax_f, d_pot, ctxv, leyenda=leyenda,
                                con_xlabel=(ax_c is None))
                        if ax_c is not None:
                            perf_ion.dibujar_iones_axial(
                                ax_c, d_ion, ctxv, curvas=curvas, leyenda=leyenda)
                        fig.tight_layout()
                        return fig

                    datos_axial = None
                    if pd is not None:
                        datos_axial = {
                            "df": pd.DataFrame({"z_nm": z_axis, "phi_V": phi_ax,
                                                "cation_mM": cp_ax, "anion_mM": cm_ax,
                                                "total_mM": ct_ax}),
                            "header": f"Central Axis Profile r=0 | {v_label} V | {stem}"}
                    st.caption(f"📍 Central axis ($r=0$) with {len(z_axis)} axial mesh nodes.")
                    panel_grafico("perf_axial", dibujar_perf_axial, carpeta_sol,
                                  f"{stem}_axial_profile_{v_label}V", datos_axial)

            # ── Promedio transversal <c>(z) ────────────────────────────────
            with tab_prom:
                with st.form("form_perf_prom", border=True):
                    curvas_prom = st.multiselect(
                        "Ion species to average and show",
                        ["Cation", "Anion", "Total ions"],
                        default=["Cation", "Anion", "Total ions"]
                    )
                    btn_prom = st.form_submit_button(f"🧮 Compute and Plot ({v_label} V)", type="primary", use_container_width=True)

                if btn_prom:
                    st.session_state["v_calc__perf_prom"] = v_label

                if st.session_state.get("v_calc__perf_prom") != v_label:
                    st.info(f"👆 Selected voltage: **{v_label} V**. Adjust the options and click **🧮 Compute and Plot ({v_label} V)**.")
                else:
                    ctxv   = ctx_v(v_label)
                    dm     = datos_malla(h5_path, mtime)
                    cp, cm = campos_mM(h5_path, mtime, v_label, c0)
                    res_prom = perfil_concentracion_promedio(
                        dm["z_nm"], dm["r_nm"], dm["tri"], cp, cm,
                        meta.get("z_tip"), meta.get("z_base"),
                        meta.get("R_tip"), meta.get("R_base"), c0)

                    if res_prom is None:
                        st.warning("⚠️ Could not compute the cross-section average (missing cone geometry metadata in `_sim.json`).")
                    else:
                        curvas = _curvas(curvas_prom)

                        def dibujar_perf_prom(leyenda, limpio):
                            return perf_prom.crear_figura(res_prom, ctxv,
                                                          curvas=curvas, leyenda=leyenda)

                        datos_prom = None
                        if pd is not None:
                            datos_prom = {
                                "df": pd.DataFrame({"z_nm": res_prom["z_nm"],
                                                    "cation_prom_mM": res_prom["cp_prom"],
                                                    "anion_prom_mM": res_prom["cm_prom"],
                                                    "total_prom_mM": res_prom["ct_prom"]}),
                                "header": f"Section Average Profile <c>(z) | {v_label} V | {stem}"}
                        st.caption(f"⭕ Area-weighted cross-section average with {len(res_prom['z_nm'])} slices along z.")
                        panel_grafico("perf_prom", dibujar_perf_prom, carpeta_sol,
                                      f"{stem}_section_avg_{v_label}V", datos_prom)

        # ── B.2: POTENCIAL 2D (+ LÍNEAS DE CAMPO) ───────────────────────────
        elif vista == sub_nombres[1]:
            with st.form("form_pot_2d", border=True):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    con_campo = st.checkbox("Overlay field lines E = −∇φ", value=True)
                with col_f2:
                    zoom_local = st.checkbox("Zoom in on channel tip", value=True)
                btn_pot = st.form_submit_button(f"🧮 Compute and Plot ({v_label} V)", type="primary", use_container_width=True)

            if btn_pot:
                st.session_state["v_calc__pot2d"] = v_label

            if st.session_state.get("v_calc__pot2d") != v_label:
                st.info(f"👆 Selected voltage: **{v_label} V**. Adjust the options and click **🧮 Compute and Plot ({v_label} V)**.")
            else:
                ctxv   = ctx_v(v_label)
                dm     = datos_malla(h5_path, mtime)
                phi_V  = extraer_campos(h5_path, mtime, v_label)["phi_adim"] * V_T
                d_pot  = {"z_nm": dm["z_nm"], "r_nm": dm["r_nm"], "tri": dm["tri"],
                          "phi_V": phi_V}

                # Campo eléctrico cacheado (evita recalcular en cada re-run)
                ec = None
                if con_campo:
                    try:
                        ec = campo_electrico(h5_path, mtime, v_label, V_T)
                    except Exception as e:
                        st.warning(f"Could not compute the field lines: {e}")

                def dibujar_potencial(leyenda, limpio):
                    # El campo E ya viene cacheado (campo_electrico); se lo
                    # pasamos al módulo para que no lo recalcule en cada re-run.
                    xlim, ylim = pc.limites_zoom_local(ctxv) if zoom_local else (None, None)
                    return potential.crear_figura(d_pot, ctxv,
                                                  con_campo=(ec is not None),
                                                  leyenda=leyenda, campo=ec,
                                                  xlim=xlim, ylim=ylim)

                panel_grafico("pot2d", dibujar_potencial, carpeta_sol,
                              f"{stem}_potential2D_{v_label}V")

        # ── B.3: MAPAS DE CONCENTRACIÓN 2D ──────────────────────────────────
        elif vista == sub_nombres[2]:
            with st.form("form_iones_2d", border=True):
                especie_mapa = st.radio(
                    "Species or view to plot",
                    [f"Cation ({sal['label_p']})",
                     f"Anion ({sal['label_m']})",
                     f"Total ions ({sal['label_p']} + {sal['label_m']})",
                     "Side-by-side comparison (Cation vs. Anion)"],
                    horizontal=True
                )
                zoom_local = st.checkbox("Zoom in on channel tip", value=True)
                btn_c2d = st.form_submit_button(f"🧮 Compute and Plot ({v_label} V)", type="primary", use_container_width=True)

            if btn_c2d:
                st.session_state["v_calc__iones2d"] = v_label

            if st.session_state.get("v_calc__iones2d") != v_label:
                st.info(f"👆 Selected voltage: **{v_label} V**. Adjust the options and click **🧮 Compute and Plot ({v_label} V)**.")
            else:
                ctxv  = ctx_v(v_label)
                datos = datos_2d(h5_path, mtime, v_label, c0)

                if "Cation" in especie_mapa and "Anion" not in especie_mapa:
                    especie, es_comp = "cation", False
                elif "Anion" in especie_mapa and "Cation" not in especie_mapa:
                    especie, es_comp = "anion", False
                elif "total ions" in especie_mapa.lower():
                    especie, es_comp = "total", False
                else:
                    especie, es_comp = None, True

                def dibujar_iones(leyenda, limpio):
                    xlim, ylim = pc.limites_zoom_local(ctxv) if zoom_local else (None, None)
                    if es_comp:
                        return ions.crear_figura_comparativo(datos, ctxv, leyenda=leyenda, xlim=xlim, ylim=ylim)
                    return ions.crear_figura(datos, ctxv, especie=especie, leyenda=leyenda, xlim=xlim, ylim=ylim)

                datos_i = None
                if pd is not None:
                    datos_i = {"df": pd.DataFrame({"z_nm": datos["z_nm"], "r_nm": datos["r_nm"],
                                                   "cation_mM": datos["cp_mM"], "anion_mM": datos["cm_mM"],
                                                   "total_mM": datos["cp_mM"] + datos["cm_mM"]}),
                               "header": f"2D nodes ({especie_mapa}) | {v_label} V | {stem}"}
                panel_grafico("iones2d", dibujar_iones, carpeta_sol,
                              f"{stem}_map2D_{especie_mapa.split(' ')[0]}_{v_label}V", datos_i)

        # ── B.4: PRECIPITACIÓN (DAVIES) — solo sales poco solubles ──────────
        elif hay_precip and vista == sub_nombres[3]:
            Kps = sal["Ksp_M2"]
            if Kps is None:
                Kps = pc.KPS_LEGACY_M2
                st.caption(f"⚠️ The _sim.json does not define Ksp: using legacy "
                           f"{pc.KPS_LEGACY_M2:.2e} M² (KClO4 at 25 °C).")

            with st.form("form_precip", border=True):
                factor = st.slider(
                    "Supersaturation factor (× Ksp)",
                    min_value=0.5, max_value=3.0, value=2.0, step=0.05,
                    help="Precipitates where Q_act > factor·Ksp. The paper uses "
                         "factor = 2 (S ≈ 1.4).")
                zoom_local = st.checkbox("Zoom in on channel tip", value=True)
                btn_precip = st.form_submit_button(f"🧮 Compute and Plot ({v_label} V)", type="primary", use_container_width=True)

            if btn_precip:
                st.session_state["v_calc__precip"] = v_label

            if st.session_state.get("v_calc__precip") != v_label:
                st.info(f"👆 Selected voltage: **{v_label} V**. Adjust the options and click **🧮 Compute and Plot ({v_label} V)**.")
            else:
                ctxv  = ctx_v(v_label)
                datos = datos_2d(h5_path, mtime, v_label, c0)

                res = precipitation.calcular_precipitacion(
                    datos["cp_mM"], datos["cm_mM"], sal["z_p"], sal["z_m"], Kps, factor)
                st.metric("Precipitating nodes",
                          f"{res['n_precip']}/{res['n_total']} "
                          f"({100.0 * res['n_precip'] / res['n_total']:.2f} %)")

                def dibujar_precip(leyenda, limpio):
                    xlim, ylim = pc.limites_zoom_local(ctxv) if zoom_local else (None, None)
                    return precipitation.crear_figura(datos, ctxv, factor=factor,
                                                      leyenda=leyenda, figsize=(11, 6),
                                                      xlim=xlim, ylim=ylim)

                panel_grafico("precip", dibujar_precip, carpeta_sol,
                              f"{stem}_precipitation_{v_label}V_f{factor:g}")
