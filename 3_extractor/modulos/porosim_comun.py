# -*- coding: utf-8 -*-
"""
porosim_comun — capa compartida del extractor (Pilar 3).

Única fuente de verdad para todo lo que ANTES estaba repetido en cada módulo
y duplicado dentro de la GUI:

  · Lectura defensiva del _sim.json (metadatos, catálogo de sal, films).
  · Carga de la malla y de las soluciones FEniCS del .h5 (con stderr
    silenciado para los warnings de HDF5 >= 1.14).
  · Detección de voltajes disponibles (rampa V_max/n_steps + catálogo fijo).
  · Espejo axisimétrico r → -r y nodos del eje r = 0.
  · Decoración estándar de las figuras: bocas del canal, films (bandas o
    rectángulos), escala divergente de potencial anclada en 0 V, limpieza
    de figura para publicación.

Los módulos de modulos/<categoria>/ lo importan con el boilerplate:

    import sys, pathlib
    _MOD = str(pathlib.Path(__file__).resolve().parents[1])
    if _MOD not in sys.path:
        sys.path.insert(0, _MOD)
    import porosim_comun as pc

y la GUI (gui_extractor_app.py) lo importa igual. Así, editar UNA función
acá repercute en el modo consola (extractor.py) Y en la GUI a la vez.

FEniCS se importa adentro de las funciones de carga (no a nivel de módulo):
importar porosim_comun es liviano y no arrastra dolfin hasta que hace falta.
"""
import os
import sys
import json
import pathlib
from contextlib import contextmanager

import numpy as np

# ─── Constantes físicas ──────────────────────────────────────────────────────
R_GAS    = 8.314          # J/(mol·K)
F_CONST  = 96485.3        # C/mol
E_CHARGE = 1.602176634e-19  # C
T_DEFAULT_K = 298.15

KPS_LEGACY_M2 = 1.07e-2   # KClO4 a 25 °C — fallback para _sim.json legacy


def voltaje_termico(T_K=T_DEFAULT_K):
    """V_T = R·T/F (~0.0257 V a 298.15 K). Des-normaliza φ: φ_V = φ_adim·V_T."""
    return R_GAS * float(T_K) / F_CONST


# =============================================================================
# ENTORNO / IO DEFENSIVO
# =============================================================================
def preparar_entorno_fenics():
    """Deshabilita MPI/UCX y limita threads ANTES de importar dolfin.
    setdefault: si el lanzador ya los seteó, no los pisa."""
    os.environ.setdefault("OMPI_MCA_pml", "^ucx")
    os.environ.setdefault("OMPI_MCA_btl", "^openib")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")


@contextmanager
def stderr_silencioso():
    """Suprime los warnings de HDF5 >= 1.14 al abrir archivos de dolfin."""
    original = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original


def _fenics():
    preparar_entorno_fenics()
    import fenics
    return fenics


# =============================================================================
# METADATOS (_sim.json hermano de la solución)
# =============================================================================
def cargar_meta(carpeta_sol):
    """Carga el *_sim.json de la carpeta de la solución ({} si falta/roto)."""
    candidatos = sorted(carpeta_sol.glob("*_sim.json"))
    if not candidatos:
        return {}
    try:
        with open(candidatos[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def info_sal(sim):
    """Bloque "salt" del catálogo del solver, con fallback legacy (campos
    planos electrolito/z_p/z_m de _sim.json viejos). Una mezcla (v1.1.0,
    "salt" = null) no tiene UNA sal: name = la etiqueta del electrolito,
    label_p/label_m = los cationes/aniones unidos con " + ", z_p/z_m = None."""
    sal = sim.get("salt")
    if isinstance(sal, dict) and sal:
        return {
            "name":  sal.get("name", "custom"),
            "label_p": sal.get("cation", {}).get("symbol", "cation"),
            "label_m": sal.get("anion", {}).get("symbol", "anion"),
            "z_p":     sal.get("cation", {}).get("z", 1),
            "z_m":     sal.get("anion", {}).get("z", -1),
            "soluble": sal.get("soluble"),
            "Ksp_M2":  sal.get("Ksp_M2"),
            "legacy":  False,
        }
    iones = sim.get("ions")
    if isinstance(iones, list) and iones:
        return {
            "name":    sim.get("electrolyte_label") or "mixture",
            "label_p": " + ".join(str(i.get("symbol")) for i in iones if i.get("z", 0) > 0),
            "label_m": " + ".join(str(i.get("symbol")) for i in iones if i.get("z", 0) < 0),
            "z_p":     None,
            "z_m":     None,
            "soluble": None,
            "Ksp_M2":  None,
            "legacy":  False,
        }
    return {
        "name":  sim.get("electrolito", "KCl"),
        "label_p": f"cation (z={sim.get('z_p', 1):+d})",
        "label_m": f"anion (z={sim.get('z_m', -1):+d})",
        "z_p":     sim.get("z_p", 1),
        "z_m":     sim.get("z_m", -1),
        "soluble": None,
        "Ksp_M2":  None,
        "legacy":  True,
    }


def iones_de_sim(sim):
    """Iones de la corrida en el ORDEN de las componentes u_i de
    U = (φ, u_1, ..., u_N): [{"symbol", "z", "D_m2s", "c0_mM"}]. Viene en
    sim["ions"] desde v1.1.0; para un _sim.json anterior (siempre binario) se
    arma con la sal y la c0 (c0 = 100 mM si falta, como contexto_de)."""
    iones = sim.get("ions")
    if isinstance(iones, list) and iones:
        return [dict(i) for i in iones]
    sal  = sim.get("salt") if isinstance(sim.get("salt"), dict) else {}
    info = info_sal(sim)
    c0   = sim.get("c0_mM", 100.0)
    c0   = 100.0 if c0 is None else float(c0)
    return [{"symbol": info["label_p"], "z": info["z_p"], "c0_mM": c0,
             "D_m2s": sal.get("cation", {}).get("D_m2s")},
            {"symbol": info["label_m"], "z": info["z_m"], "c0_mM": c0,
             "D_m2s": sal.get("anion", {}).get("D_m2s")}]


def es_binaria(sim):
    """¿Es el caso de 2 iones con UNA c0 que suponen los módulos de
    concentración (c± = c0·exp(u±))? False para una mezcla o una sal
    asimétrica (el solver deja c0_mM = null en ambos casos)."""
    return sim.get("c0_mM", 100.0) is not None and len(iones_de_sim(sim)) == 2


def describir_electrolito(sim):
    """'KCl · 100 mM' (binario) o 'KCl_100mM+HCl_10mM: K⁺ 100, Cl⁻ 110, H⁺ 10 mM'."""
    if es_binaria(sim):
        return f"{info_sal(sim)['name']} · {float(sim.get('c0_mM', 100.0)):g} mM"
    iones = iones_de_sim(sim)
    nombre = sim.get("electrolyte_label") or info_sal(sim)["name"]
    return (f"{nombre}: " +
            ", ".join(f"{i.get('symbol')} {float(i.get('c0_mM')):g}" for i in iones) + " mM")


SOLO_BINARIA = ("this module draws the cation/anion pair of a binary electrolyte. "
                "For a mixture (or an asymmetric salt) use the potential map, "
                "the field lines, the axial potential profile and the I-V.")


def aplica_binaria(meta):
    """aplica() de los módulos de concentración (mapas de iones, perfiles de
    iones, precipitación): solo electrolitos binarios."""
    return es_binaria(meta.get("simulation", {}))


def exigir_binaria(meta):
    """Para el preparar() de esos módulos: True si la corrida es binaria con
    c0 conocida; si no, imprime el motivo y devuelve False."""
    sim = meta.get("simulation", {})
    if sim.get("ions") and not es_binaria(sim):
        print(f"❌ {describir_electrolito(sim)}: {SOLO_BINARIA}")
        return False
    if sim.get("c0_mM") is None:
        print("❌ Missing 'c0_mM' in the _sim.json: the concentration maps "
              "are not reliable.")
        return False
    return True


def films_activos_de(meta):
    """Films activos con sus bordes en z (m). Cada film ocupa de la interfaz
    film/agua (z_film_<lado>) a la boca de la membrana (z_tip / z_base)."""
    z_boca = {"tip": meta.get("z_tip"), "base": meta.get("z_base")}
    films = []
    for film_dict in meta.get("simulation", {}).get("films", []):
        lado   = film_dict.get("side")
        z_int  = meta.get(f"z_film_{lado}")
        z_memb = z_boca.get(lado)
        if z_int is None or z_memb is None:
            continue
        films.append({
            "side":     lado,
            "z_int":    z_int,
            "z_memb":   z_memb,
            "type":     film_dict.get("type", "?"),
            "rho":      film_dict.get("rho_fix_target_Cm3", 0.0),
            "phi_D_mV": film_dict.get("phi_D_anal_mV"),
        })
    return films


def contexto_de(meta, stem, v_label=None):
    """Contexto estándar que reciben las crear_figura() de los módulos.
    Lo construyen igual el procesar() de consola y la GUI: misma figura.

    Claves: stem, v_label, v_num, c0 (mM), T_K, V_T (V), sal (dict de
    info_sal), z_tip/z_base/R_tip/R_base (m), films, sigma_Cm2, ions (lista
    de iones_de_sim) y binaria (es_binaria). En una mezcla c0 = None: los
    módulos de concentración no aplican (su aplica() lo dice)."""
    sim = meta.get("simulation", {})
    sal = info_sal(sim)
    T_K = float(sim.get("T_K", T_DEFAULT_K))
    c0  = sim.get("c0_mM", 100.0)
    return {
        "stem":      stem,
        "v_label":   v_label,
        "v_num":     float(v_label) if v_label is not None else None,
        "c0":        float(c0) if c0 is not None else None,
        "ions":      iones_de_sim(sim),
        "binaria":   es_binaria(sim),
        "T_K":       T_K,
        "V_T":       voltaje_termico(T_K),
        "salt":       sal,
        "z_tip":     meta.get("z_tip"),
        "z_base":    meta.get("z_base"),
        "R_tip":     meta.get("R_tip"),
        "R_base":    meta.get("R_base"),
        "L_charge":  meta.get("L_charge", 0.0),
        "films":     films_activos_de(meta),
        "sigma_Cm2": sim.get("sigma_Cm2"),
    }


# =============================================================================
# CARGA FEniCS (malla / voltajes / campos nodales)
# =============================================================================
def cargar_malla(ruta_h5):
    """Lee /malla del .h5 del solver y devuelve el Mesh."""
    fe = _fenics()
    mesh = fe.Mesh()
    with stderr_silencioso():
        hdf = fe.HDF5File(mesh.mpi_comm(), str(ruta_h5), "r")
        try:
            hdf.read(mesh, "/malla", False)
        finally:
            hdf.close()
    return mesh


def espacio_mixto(mesh, n_iones=2):
    """FunctionSpace mixto (φ, u_1, ..., u_N) — el mismo del solver. Con
    n_iones=2 es el P1×P1×P1 (φ, up, um) de siempre."""
    fe = _fenics()
    P1 = fe.FiniteElement("P", mesh.ufl_cell(), 1)
    return fe.FunctionSpace(mesh, fe.MixedElement([P1] * (1 + int(n_iones))))


def detectar_voltajes(ruta_h5, mesh, sim=None):
    """Etiquetas '+0.50' de los datasets /U_..V presentes en el .h5,
    ordenadas numéricamente. Candidatos = rampa V_max/n_steps del _sim.json
    (si está) + catálogo fijo 0.1..10 V (compatibilidad con corridas viejas)."""
    fe = _fenics()
    sim = sim or {}
    candidatos = {0.0}
    V_max, n_steps = sim.get("V_max_V"), sim.get("n_steps")
    if V_max and n_steps:
        for rama in (np.linspace(0.0, V_max, int(n_steps)),
                     np.linspace(0.0, -V_max, int(n_steps))):
            candidatos.update(round(v, 4) for v in rama)
    for v in np.arange(0.1, 10.01, 0.1):
        candidatos.update((round(v, 4), round(-v, 4)))

    por_label = {f"{v:+.2f}": v for v in candidatos}     # dedupe por etiqueta
    with stderr_silencioso():
        hdf = fe.HDF5File(mesh.mpi_comm(), str(ruta_h5), "r")
        try:
            labels = [lab for lab in por_label if hdf.has_dataset(f"/U_{lab}V")]
        finally:
            hdf.close()
    return sorted(labels, key=float)


def normalizar_v_label(texto):
    """'-1' / '-1.0' / '' → etiqueta '+X.XX' del dataset ('' = -1.00).
    Devuelve None si no es un número."""
    texto = (texto or "").strip()
    if not texto:
        texto = "-1.00"
    try:
        return f"{float(texto):+.2f}"
    except ValueError:
        return None


def leer_campos(ruta_h5, mesh, V, v_label):
    """Arrays NumPy en los vértices para un voltaje: φ adimensional y los
    log-potenciales de cada ion, "us" = [u_1, ..., u_N] (c_i = c0_i·exp(u_i)).
    Con 2 iones también "up"/"um" (los nombres de siempre). El HDF5File se
    cierra siempre. V tiene que tener 1 + N campos (espacio_mixto)."""
    fe = _fenics()
    U = fe.Function(V)
    with stderr_silencioso():
        hdf = fe.HDF5File(mesh.mpi_comm(), str(ruta_h5), "r")
        try:
            hdf.read(U, f"/U_{v_label}V")
        finally:
            hdf.close()
    phi, *us = U.split(deepcopy=True)
    campos = {"phi_adim": phi.compute_vertex_values(mesh),
              "us":       [u.compute_vertex_values(mesh) for u in us]}
    if len(us) == 2:
        campos["up"], campos["um"] = campos["us"]
    return campos


def datos_malla(mesh):
    """Estructuras serializables de la malla: coordenadas [nm], triángulos y
    los índices de los nodos del eje r=0 ordenados por z (perfiles axiales)."""
    coords = mesh.coordinates()
    axis   = np.where(coords[:, 1] < 1e-12)[0]           # nodos sobre r = 0
    axis   = axis[np.argsort(coords[axis, 0])]           # ordenados por z
    return {"z_nm": coords[:, 0] * 1e9, "r_nm": coords[:, 1] * 1e9,
            "tri": np.array(mesh.cells()), "axis_idx": axis}


# =============================================================================
# CARGA COMPLETA + TRONCO COMÚN DE LOS preparar() DE LOS MÓDULOS
# (contrato de módulos: ver modulos/MODULE_CONTRACT.md)
# =============================================================================
def cargar_solucion(ruta_h5):
    """Toda la carga PESADA de una solución, UNA sola vez: metadatos, malla,
    espacio mixto, estructuras de graficado y voltajes disponibles.

    El modo batch la llama una vez y pasa el dict a los preparar() de los
    módulos (parámetro sol=): así un barrido --voltaje todos no recarga la
    malla por cada voltaje. Con sol=None, preparar_comun() la llama acá
    (consola: un módulo, un voltaje)."""
    ruta_h5 = pathlib.Path(ruta_h5)
    meta = cargar_meta(ruta_h5.parent)
    mesh = cargar_malla(ruta_h5)
    n_iones = len(iones_de_sim(meta.get("simulation", {})))
    return {"ruta":   ruta_h5,
            "meta":   meta,
            "mesh":   mesh,
            "V":      espacio_mixto(mesh, n_iones),
            "dm":     datos_malla(mesh),
            "labels": detectar_voltajes(ruta_h5, mesh,
                                        meta.get("simulation", {}))}


def elegir_voltaje(labels, v_label=None):
    """Etiqueta '+X.XX' validada contra los datasets del .h5.
    v_label=None → modo interactivo: lista los voltajes y pregunta (consola).
    v_label dado (str o float) → solo normaliza y valida (batch / GUI).
    Devuelve None (con aviso impreso) si es inválido o no está en el .h5."""
    if v_label is None:
        print(f"    Voltages: {[l + 'V' for l in labels]}")
        v_label = input("\nWhich voltage to visualize? [Enter = -1.00]: ")
    etiqueta = normalizar_v_label(str(v_label))
    if etiqueta is None or etiqueta not in labels:
        print(f"❌ Dataset '/U_{etiqueta}V' not found. "
              f"Disponibles: {[l + 'V' for l in labels]}")
        return None
    return etiqueta


def preparar_comun(ruta_solucion, titulo, v_label=None, sol=None):
    """Tronco común de los preparar() de los módulos: banner + carga (o reuso
    de `sol`) + contexto + elección/validación de voltaje + campos nodales de
    ese voltaje.  →  (sol, ctx, campos) o None si algo falla / se cancela.

    ctx sale con v_label/v_num ya seteados: los módulos solo transforman
    `campos` + sol["dm"] en su dict `datos` y devuelven (datos, ctx)."""
    ruta_solucion = pathlib.Path(ruta_solucion)
    print("\n" + "="*60)
    print(f"   MÓDULO: {titulo}")
    print("="*60)

    if sol is None:
        print(f">>> Loading mesh from: {ruta_solucion.name}")
        try:
            sol = cargar_solucion(ruta_solucion)
        except Exception as e:
            print(f"❌ Error loading the mesh/solution: {e}")
            return None
    if not sol["meta"]:
        print("❌ _sim.json not found in the solution folder.")
        return None
    if not sol["labels"]:
        print("❌ No solution datasets found in the file.")
        return None

    ctx = contexto_de(sol["meta"], ruta_solucion.stem)

    # Info estándar de la corrida (lo que antes imprimía cada módulo).
    if ctx["z_tip"] is not None and ctx["z_base"] is not None:
        print(f"  Channel: [{ctx['z_tip']*1e9:.1f}, {ctx['z_base']*1e9:.1f}] nm")
    for film in ctx["films"]:
        print(f"  [Film {film['side']}] type={film['type']}, "
              f"[{min(film['z_int'], film['z_memb'])*1e9:.1f}, "
              f"{max(film['z_int'], film['z_memb'])*1e9:.1f}] nm")
    if not ctx["films"]:
        print("  [Films] Geometry with NO films")
    print(f"  Electrolyte: {describir_electrolito(sol['meta'].get('simulation', {}))}")

    etiqueta = elegir_voltaje(sol["labels"], v_label)
    if etiqueta is None:
        return None
    ctx["v_label"], ctx["v_num"] = etiqueta, float(etiqueta)

    print(f">>> Extracting: /U_{etiqueta}V")
    try:
        campos = leer_campos(sol["ruta"], sol["mesh"], sol["V"], etiqueta)
    except Exception as e:
        print(f"❌ Error loading the solution: {e}")
        return None
    return sol, ctx, campos


# =============================================================================
# GEOMETRÍA DE GRAFICADO
# =============================================================================
def limites_zoom_local(ctx):
    """Calcula xlim y ylim para hacer zoom dinámico en el tip del canal,
    igual que en el mesher."""
    z_tip = ctx.get("z_tip")
    if z_tip is None:
        return None, None
        
    delta_tip = 0.0
    for film in ctx.get("films", []):
        if film["side"] == "tip":
            delta_tip = abs(film["z_int"] - film["z_memb"])
            
    z_film_tip = (z_tip - delta_tip) * 1e9
    z_tip_nm = z_tip * 1e9
    r_tip_nm = ctx.get("R_tip", 0.0) * 1e9
    L_charge_nm = ctx.get("L_charge", 0.0) * 1e9
    
    margin_x = max(150.0, r_tip_nm * 3.0, L_charge_nm * 1.5)
    xlim = (z_film_tip - margin_x * 0.5, z_tip_nm + margin_x)
    
    margin_y = max(50.0, r_tip_nm * 0.5)
    r_max = r_tip_nm + L_charge_nm + margin_y
    ylim = (-r_max, r_max)
    
    return xlim, ylim


def espejo(z_nm, r_nm, tri, campo):
    """Espejo axisimétrico r → -r para ver el poro completo en los mapas 2D."""
    n = len(r_nm)
    return (np.concatenate([z_nm, z_nm]),
            np.concatenate([r_nm, -r_nm]),
            np.concatenate([tri, tri + n]),
            np.concatenate([campo, campo]))


# =============================================================================
# DECORACIÓN ESTÁNDAR DE FIGURAS
# =============================================================================
def guias_canal(ax, ctx, con_label=False):
    """Channel mouths: líneas verticales verdes punteadas en z_tip y z_base."""
    primera = True
    for z in (ctx.get("z_tip"), ctx.get("z_base")):
        if z is not None:
            ax.axvline(z * 1e9, color="green", ls="--", lw=1.0, alpha=0.5,
                       label=("Channel mouths" if con_label and primera else None))
            primera = False


def bandas_films(ax, ctx, con_labels=True):
    """Films como banda vertical: roja (carga +) o azul (carga −).
    Estilo de los perfiles 1D y del mapa de precipitación."""
    for film in ctx.get("films", []):
        z0 = min(film["z_int"], film["z_memb"]) * 1e9
        z1 = max(film["z_int"], film["z_memb"]) * 1e9
        color = "red" if film["rho"] > 0 else "blue"
        signo = "+" if film["rho"] > 0 else "−"
        label = f"Film {film['side']} (carga {signo})" if con_labels else None
        ax.axvspan(z0, z1, color=color, alpha=0.12, label=label)


def rectangulos_films(ax, ctx, r_max_nm, r_min_nm=None, con_labels=True):
    """Films como rectángulo SIN relleno (borde rojo/azul): estilo de los
    mapas 2D, donde el interior lleva la información del campo graficado.
    r_min_nm=None → rectángulo simétrico (-r_max, +r_max)."""
    import matplotlib.patches as mpatches
    if r_min_nm is None:
        r_min_nm = -r_max_nm
    for film in ctx.get("films", []):
        z0 = min(film["z_int"], film["z_memb"]) * 1e9
        z1 = max(film["z_int"], film["z_memb"]) * 1e9
        color = "red" if film["rho"] > 0 else "blue"
        signo = "+" if film["rho"] > 0 else "−"
        label = (f"Film {film['side']} ({film['type']}, charge {signo})"
                 if con_labels else None)
        
        # Efecto de alto contraste: borde blanco sólido de fondo (finito)
        ax.add_patch(mpatches.Rectangle(
            (z0, r_min_nm), z1 - z0, r_max_nm - r_min_nm,
            linewidth=1.2, edgecolor="white", facecolor="none"))
            
        # Línea de color por encima (fina y punteada)
        ax.add_patch(mpatches.Rectangle(
            (z0, r_min_nm), z1 - z0, r_max_nm - r_min_nm,
            linewidth=0.8, edgecolor=color, linestyle="--", facecolor="none", label=label))


def escala_potencial(phi_V, V_T=None):
    """(norm, niveles) para los mapas de φ: escala divergente RdBu_r con el
    BLANCO SIEMPRE anclado en 0 V (el outlet está a 0 V y debe verse neutro).
    El padding mínimo (eps) permite TwoSlopeNorm aunque el rango no cruce 0."""
    import matplotlib.colors as mcolors
    if V_T is None:
        V_T = voltaje_termico()
    eps    = max(np.abs(phi_V).max(), V_T) * 1e-3
    phi_lo = min(phi_V.min(), -eps)
    phi_hi = max(phi_V.max(),  eps)
    norm    = mcolors.TwoSlopeNorm(vmin=phi_lo, vcenter=0.0, vmax=phi_hi)
    niveles = np.linspace(phi_lo, phi_hi, 100)
    return norm, niveles


def marcar_colorbar(cbar):
    """Marca la colorbar para que limpiar_figura() la preserve."""
    cbar.ax._es_colorbar = True
    return cbar


def ticks_colorbar(cbar, vmin, vmax, nbins=10):
    """~nbins ticks en múltiplos redondos, descartando los que caen fuera de
    [vmin, vmax]. MaxNLocator devuelve múltiplos ALREDEDOR del intervalo sin
    recortarlos y matplotlib dibuja los de afuera pegados al borde de la
    barra, encimados con el tick vecino (números ilegibles)."""
    from matplotlib.ticker import MaxNLocator
    ticks = MaxNLocator(nbins=nbins).tick_values(vmin, vmax)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    cbar.set_ticks(ticks)
    return ticks


def limpiar_figura(fig):
    """Imagen limpia para publicaciones: sin título, labels de ejes, números
    de ticks ni leyenda, pero con el marco y las marcas de los ticks en los 4
    lados (la convención usual: la imagen queda limpia sin perder la
    referencia de escala). Las colorbars marcadas con marcar_colorbar() se
    preservan; el fondo (patch) también, para que el gris de la membrana
    quede en la exportación."""
    fig.suptitle("")
    for ax in fig.axes:
        if getattr(ax, "_es_colorbar", False):
            continue
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(top=True, right=True)
        ax.grid(False)
        leyenda = ax.get_legend()
        if leyenda is not None:
            leyenda.remove()
    return fig


# =============================================================================
# GUARDADO ESTÁNDAR (contrato guardar() de los módulos)
# =============================================================================
def guardar_figura(fig, ruta_solucion, nombre, limpio=False, dpi=300):
    """savefig estándar del extractor en la carpeta de la solución
    (dpi 300, bbox tight) + close. limpio=True → limpiar_figura() antes
    (mapas publication-ready, como el '¿Guardar mapa limpio?' de consola).
    Devuelve el Path guardado."""
    import matplotlib.pyplot as plt
    if limpio:
        limpiar_figura(fig)
    fpath = pathlib.Path(ruta_solucion).parent / nombre
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    ✓ Saved: {fpath}")
    return fpath


def exportar_txt(ruta_solucion, nombre, columnas, header):
    """np.savetxt estándar (%.6e, tab) en la carpeta de la solución.
    columnas = lista de arrays 1D del mismo largo. Devuelve el Path."""
    fpath = pathlib.Path(ruta_solucion).parent / nombre
    np.savetxt(fpath, np.column_stack(columnas), header=header,
               fmt="%.6e", delimiter="\t")
    print(f"    ✓ Saved: {fpath}")
    return fpath


def exportar_xlsx(ruta_solucion, nombre, columnas_dict):
    """DataFrame → .xlsx en la carpeta de la solución. Devuelve el Path, o
    None si faltan pandas/openpyxl (avisa y sigue: el .txt ya salió)."""
    try:
        import pandas as pd
        fpath = pathlib.Path(ruta_solucion).parent / nombre
        pd.DataFrame(columnas_dict).to_excel(fpath, index=False)
    except ImportError:
        print("  ❌ Missing pandas/openpyxl for the .xlsx "
              "(pip install pandas openpyxl). Se omite.")
        return None
    print(f"    ✓ Saved: {fpath}")
    return fpath
