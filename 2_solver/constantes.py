# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — SOLVER · CONSTANTES (física universal + perillas + catálogos)
  Secciones 1-3 del solver original del paper, transcriptas SIN CAMBIOS.
  Sin FEniCS: este módulo lo importan también la ayuda y las preguntas.
═══════════════════════════════════════════════════════════════════════════
"""
import os
import numpy as np

# =============================================================================
# SECCIÓN 1: CONSTANTES FÍSICAS UNIVERSALES  (NO TOCAR)
# =============================================================================
# Constantes de la naturaleza. No dependen del experimento ni del usuario.
R_GAS          = 8.314          # constante de los gases   [J/(mol·K)]
F_CONST        = 96485.3        # constante de Faraday     [C/mol]
EPS_0          = 8.854e-12      # permitividad del vacío   [F/m]
E_CHARGE       = 1.602176634e-19  # carga elemental        [C]

# Factor de conversión entre densidad de carga del film expresada como
# concentración molar y como densidad numérica de electrones por nm³:
#     n_e [e/nm³] = c_fix [M] / E_NM3_TO_MOLAR
#     (1 e/nm³ = 1e27 / N_A mol/m³ = 1660.6 mol/m³ = 1.6606 M)
E_NM3_TO_MOLAR = 1.6606

# Factores de post-procesamiento de corriente
FACTOR_AXISIM  = 2.0 * np.pi    # revolución del semiplano (∫ ... 2π r ds)
A_TO_NA        = 1e9            # conversión A → nA

# =============================================================================
# SECCIÓN 2: PERILLAS NUMÉRICAS DEL SOLVER  (tocar solo si no converge)
# =============================================================================
# No son física ni elección del usuario: están afinadas para que Newton
# converja. Cambiarlas solo ante problemas de convergencia.

LINEAR_SOLVER     = 'mumps'     # solver lineal directo robusto
NEWTON_MAX_ITER   = 25          # tope de iteraciones de Newton
NEWTON_REPORT     = True        # imprimir progreso de Newton
GRADO_ELEMENTO    = 1           # grado del elemento finito (P1)

# Tolerancias de Newton (absoluta, relativa) según etapa:
TOL_EQUILIBRIO    = (1e-20, 1e-8)   # rampas de σ y de ρ_film (estrictas)
TOL_VOLTAJE       = (1e-12, 1e-8)   # barrido de voltaje (relajadas)

# Pasos de las rampas:
# N_STEPS_FILM es el default; se puede subir por corrida desde el params.json
# (campo "n_steps_film"). Receta validada para film muy cargado + electrolito
# diluido (contraste c_fix/c0 grande): rampas MÁS LENTAS — más n_steps de
# voltaje y/o más n_steps_film (ver README, limitaciones conocidas).
N_STEPS_FILM      = 40          # pasos de la rampa de carga volumétrica del film
MAX_SUBDIVISIONES = 6           # subdivisiones máximas del paso de voltaje

# =============================================================================
# SECCIÓN 3: CATÁLOGO DE TIPOS DE FILM  (derivado del factor de conversión)
# =============================================================================
# Tipos de film predefinidos, expresados en MOLAR de carga fija.
# Los valores en e/nm³ se DERIVAN del factor (no se hardcodean), de modo que
# el factor vive en un solo lugar (E_NM3_TO_MOLAR) y no puede desincronizarse.
#
# ⚠ Bug histórico (2026-05-26): versiones viejas hardcodeaban n_e = valor_molar
#   directamente (ej. "2M" usaba n_e=2.0 → c_fix=3.32M en vez de 2M). Resuelto
#   estructuralmente: ahora SIEMPRE se calcula como molar / E_NM3_TO_MOLAR.
FILM_TIPOS_MOLAR = [0.66, 1.0, 2.0, 4.0]
def _fmt_molar(m):
    """Formatea el valor molar como clave: 1.0 → '1M', 0.66 → '0.66M'."""
    return (f"{m:g}M")   # %g elimina ceros/punto decimal innecesarios
film_tipos = {_fmt_molar(m): m / E_NM3_TO_MOLAR for m in FILM_TIPOS_MOLAR}

# Defaults de la corrida (se ofrecen como valor por defecto en las preguntas
# y son los defaults de los campos opcionales del params.json)
C0_DEFAULT_MM       = 100.0     # concentración bulk
V_MAX_DEFAULT_V     = 1.0       # voltaje máximo del barrido
N_STEPS_DEFAULT     = 11        # puntos por rama (incl. 0 V)
T_DEFAULT_K         = 298.15    # temperatura
EPS_R_DEFAULT       = 80.0      # constante dieléctrica del agua
SIGMA_DEFAULT_E_NM2 = -1.0      # carga superficial en e/nm²
FILM_TIPO_DEFAULT   = "4M"      # tipo de film por defecto

# Catálogo de sales: fuente única de verdad de las propiedades del electrolito
# (valencias, difusividades, solubilidad, Kps). Vive en un .json junto al solver
# para que agregar una sal sea agregar una entrada, sin tocar código.
SALES_JSON  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales.json")
SAL_DEFAULT = "KCl"             # sal por defecto del catálogo

# σ por defecto en C/m² (derivado de e/nm²):
#   1 e/nm² = E_CHARGE / (1e-9)² C/m² = 1.602e-19 / 1e-18 = 0.1602 C/m²
SIGMA_DEFAULT_CM2 = SIGMA_DEFAULT_E_NM2 * E_CHARGE / (1e-9)**2   # = -0.1602

# Raíz del proyecto (la carpeta que contiene 2_solver/): ahí viven
# RESULTS/meshes, RESULTS/solutions, RESULTS/equilibria y el
# historial de la última malla usada.
REPO_ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_MALLAS     = os.path.join(REPO_ROOT, "RESULTS", "meshes")
RESULTADOS_SOLUCIONES = os.path.join(REPO_ROOT, "RESULTS", "solutions")
EQ_DIR                = os.path.join(REPO_ROOT, "RESULTS", "equilibria")
HISTORIAL_MALLA       = os.path.join(REPO_ROOT, "RESULTS", ".ultima_malla.txt")


def cargar_catalogo_sales(ruta=SALES_JSON):
    """
    Carga el catálogo de sales (sales.json) y devuelve el dict 'salts'
    (clave = nombre de la sal → propiedades). Aborta con mensaje claro si el
    archivo no está: el solver necesita el catálogo para conocer valencias,
    difusividades, solubilidad y Kps de la sal elegida.
    """
    import sys, json
    if not os.path.exists(ruta):
        print(f"\n[ERROR] Salt catalog not found: {ruta}")
        print("        The solver requires sales.json (next to the solver) to")
        print("        know the electrolyte properties.")
        sys.exit(1)
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["salts"]


def derivar_malla(ruta_archivo):
    """De la ruta a un archivo de la malla devuelve (INPUT_DIR, m_name)."""
    carpeta = os.path.dirname(ruta_archivo)
    base    = os.path.basename(ruta_archivo)
    for suf in ("_limits.json", "_domain.xdmf", "_facets.xdmf", ".msh"):
        if base.endswith(suf):
            return carpeta, base[:-len(suf)]
    return carpeta, os.path.splitext(base)[0]


def buscar_checkpoint_compatible(parcial, eq_dir=EQ_DIR):
    """
    Busca en eq_dir un checkpoint de equilibrio cuyos metadatos coincidan con
    TODAS las claves de `parcial` (comparación por subconjunto). Devuelve la
    ruta del .h5 o None.

    El motor guarda la clave completa (incluye n_nodos/n_celdas, que requieren
    cargar la malla); la GUI usa este helper con las claves que puede conocer
    SIN cargar nada (m_name, input_dir, c0, sigma, coronas, T, eps_r, z±,
    films con su rho_target REAL en C/m³). Es un pre-chequeo informativo: la
    verificación final la hace siempre el motor con la clave completa.

    OJO: en films, rho_target debe venir en C/m³ (n_e·E_CHARGE·1e27), NO en
    e/nm³ — y el nombre de la clave del flag de coronas es "rings" (como lo
    guarda el motor), no "apply_charge_rings".
    """
    import json
    if not os.path.isdir(eq_dir):
        return None
    objetivo = json.loads(json.dumps(parcial))   # normalizar tipos como JSON
    for fn in sorted(os.listdir(eq_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(eq_dir, fn), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        h5p = os.path.join(eq_dir, fn[:-5] + ".h5")
        if os.path.exists(h5p) and all(meta.get(k) == v for k, v in objetivo.items()):
            return h5p
    return None
