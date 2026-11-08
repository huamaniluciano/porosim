# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — SOLVER · MOTOR PNP (física del paper, transcripta SIN CAMBIOS)
  Resuelve Poisson-Nernst-Planck fuertemente acoplado en 2D axisimétrico.
═══════════════════════════════════════════════════════════════════════════

La función resolver(cfg) contiene la SECCIÓN 6 del solver original (carga
malla → arma física → resuelve → exporta) más el checkpoint de equilibrio.
Las fórmulas, tolerancias y el orden de las etapas son IDÉNTICOS al solver
que generó los resultados del paper; solo cambió el empaquetado: los valores
que antes llegaban por prompts ahora llegan en el dict `cfg`.

v1.1.0: N iones (mezclas, ej. KCl + HCl). U = (φ, u_1, ..., u_N) con
u_i = ln(c_i/c0_i); cada ion tiene su ecuación de Nernst-Planck y todos entran
en la ρ de Poisson. Con una sal simétrica (z+ = −z−) las expresiones, los
nombres de archivo, la clave del checkpoint y el _sim.json son los de v1.0.0.
Qué iones se resuelven lo decide electrolito.py (sin FEniCS).

cfg (todas las claves resueltas; lo arman preguntas.py o solver.py-batch):
    input_dir, m_name          carpeta y nombre base de la malla
    T, eps_r                   condiciones del medio
    El electrolito llega de UNA de dos formas (ver electrolito.iones_de_cfg):
      · binario (paper):
        salt                   dict completo (schema de sales.json)
        c0_mM                  concentración bulk de la sal [mM] (simétrica)
      · mezcla de N iones:
        ions                   [{"symbol", "z", "D_m2s", "c0_mM"}] (c0 de cada
                               ion en el bulk; Σ z·c0 = 0)
        electrolyte            [{"salt", "c0_mM"}] (informativo, va al _sim.json)
        electrolyte_label      etiqueta de los archivos ("KCl_100mM+HCl_10mM")
    sigma_Cm2                  carga superficial [C/m²]
    aplicar_carga_coronas      bool
    films                      [{"name": lado, "type": str,
                                 "n_e_per_nm3": float}] en ORDEN DE RAMPA
                               (solo los que existen en la malla)
    V_max_V, n_steps           barrido de voltaje
    guardar_todas_sol          bool (False = solo múltiplos de 0.1 V)
    n_steps_film               pasos de la rampa de ρ_film (default 40)
    reusar_equilibrio          "auto" | True | False | "ask"
    output_dir                 carpeta de salida (se crea si no existe)

Etapas: rampa de σ → rampa de ρ_film (secuencial por film) → barrido de
voltaje adaptativo. Salidas: Solutions_*.h5 + IV_curve_*.txt + *_sim.json.
"""

from fenics import *
import numpy as np
import json
import os
import hashlib

from constantes import (
    R_GAS, F_CONST, EPS_0, E_CHARGE, FACTOR_AXISIM, A_TO_NA,
    LINEAR_SOLVER, NEWTON_MAX_ITER, NEWTON_REPORT, GRADO_ELEMENTO,
    TOL_EQUILIBRIO, TOL_VOLTAJE, N_STEPS_FILM, MAX_SUBDIVISIONES,
    NEWTON_MIN_ITER_VOLTAJE, EQ_DIR,
)
from electrolito import iones_de_cfg, donnan_psi


# =============================================================================
# AUXILIARES DEL MOTOR (verbatim del solver original)
# =============================================================================
def rampa_adaptativa(solver_nls, parametro_obj, valores, tol_abs, tol_rel,
                     tag_desc="Rampa"):
    """
    Rampa de un parámetro físico (σ, ρ_film, ...) recorriendo `valores`.
    Fija las tolerancias de Newton y resuelve en cada paso.

    Args:
        solver_nls:    NonlinearVariationalSolver configurado
        parametro_obj: objeto Constant a variar
        valores:       lista de valores a recorrer
        tol_abs:       tolerancia absoluta de Newton
        tol_rel:       tolerancia relativa de Newton
        tag_desc:      descripción para imprimir
    """
    solver_nls.parameters['newton_solver']['absolute_tolerance'] = tol_abs
    solver_nls.parameters['newton_solver']['relative_tolerance'] = tol_rel

    print(f"  [{tag_desc}] Ramp with abs_tol={tol_abs}, rel_tol={tol_rel}")

    for i, val in enumerate(valores):
        parametro_obj.assign(val)
        try:
            solver_nls.solve()
            print(f"    step {i+1:2d}/{len(valores)}  value = {val:.6e}")
        except RuntimeError as e:
            print(f"    [FAILED] step {i+1}/{len(valores)}: {str(e)[:60]}")
            raise


def construir_films_activos(meta_geo, tags):
    """
    Construye la lista de films activos a partir del JSON del mallador.

    Cada lado posible (tip, base) se incluye solo si su flag include_film_<lado>
    está en True en el JSON. Devuelve una lista de dicts; cada dict reúne TODO
    lo de ese film (tags, geometría, y placeholders para los valores físicos que
    se completan con la elección del tipo).

    Lista vacía  → caso SIN film.
    Un elemento  → un film.
    Dos elementos → ambos films (orden de la lista = orden de definición; el
                    orden de RAMPA lo fija cfg["films"]).

    Convención de naming en el JSON (debe coincidir con el mallador):
        include_film_<lado>   bool
        delta_film_<lado>     espesor [m]
        z_film_<lado>         interfaz film/agua [m]
        tag DOMAIN_FILM_<LADO>      subdominio del film
        tag FILM_<LADO>_INTERFACE  línea interfaz film/agua
    """
    films = []
    for lado in ["tip", "base"]:
        if not meta_geo.get(f"include_film_{lado}", False):
            continue
        films.append({
            "name":        lado,
            "tag_domain":    tags[f"DOMAIN_FILM_{lado.upper()}"],
            "tag_interface": tags[f"FILM_{lado.upper()}_INTERFACE"],
            "delta":         meta_geo[f"delta_film_{lado}"],
            "z_interface":   meta_geo[f"z_film_{lado}"],
            # --- placeholders, se completan con la elección del tipo ---
            "type":          None,            # etiqueta ("4M", "custom_3.00M", ...)
            "n_e_per_nm3":   None,            # densidad de carga [e/nm³]
            "rho_target":    None,            # densidad de carga fija objetivo [C/m³]
            "rho_fix":       Constant(0.0),   # Constant que se rampa de 0 a rho_target
            "c_fix_eq":      None,            # [mol/m³]
            "phi_D_anal_mV": None,            # Donnan analítico [mV]
            "cK_film_anal":  None,            # [mol/m³] (catión, modo binario)
            "cCl_film_anal": None,            # [mol/m³] (anión, modo binario)
            "c_film_anal":   None,            # {símbolo: [mol/m³]} (modo general)
        })
    return films


def _preguntar_ok(texto):
    """Sí/no por consola (solo para reusar_equilibrio='preguntar')."""
    while True:
        r = input(f"{texto} [Enter=Yes / n=change]: ").strip().lower()
        if r in ['', 's', 'si', 'y', 'yes']:
            return True
        elif r in ['n', 'no']:
            return False
        else:
            print("    [Please answer 'y' (for yes/enter) or 'n' (for no/change)]")


# =============================================================================
# MOTOR — carga malla → arma física → resuelve → exporta
# =============================================================================
def resolver(cfg):
    """Corre la simulación completa según cfg. Devuelve dict con las rutas de
    salida y la curva IV. La física es la del solver original del paper."""

    INPUT_DIR  = cfg["input_dir"]
    m_name     = cfg["m_name"]
    OUTPUT_DIR = cfg["output_dir"]
    T          = cfg["T"]
    eps_r      = cfg["eps_r"]
    sal        = cfg.get("salt")        # None en modo mezcla
    c0_mM      = cfg.get("c0_mM")       # None en modo mezcla
    sigma_Cm2  = cfg["sigma_Cm2"]
    aplicar_carga_coronas = cfg["apply_charge_rings"]
    V_max_V    = cfg["V_max_V"]
    n_steps    = cfg["n_steps"]
    guardar_todas_sol = cfg["guardar_todas_sol"]
    n_steps_film      = cfg.get("n_steps_film", N_STEPS_FILM)
    reusar_equilibrio = cfg.get("reuse_equilibrium", "ask")

    # Perillas numéricas de las rampas (casi nunca se tocan; los defaults son
    # EXACTAMENTE los del solver del paper: 10 pasos de σ y TOL_EQUILIBRIO en
    # ambas rampas). Se exponen por cfg para la GUI/batch avanzados.
    n_steps_sigma = int(cfg.get("n_steps_sigma", 10))
    tol_sigma = (float(cfg.get("tol_sigma_abs", TOL_EQUILIBRIO[0])),
                 float(cfg.get("tol_sigma_rel", TOL_EQUILIBRIO[1])))
    tol_film  = (float(cfg.get("tol_film_abs",  TOL_EQUILIBRIO[0])),
                 float(cfg.get("tol_film_rel",  TOL_EQUILIBRIO[1])))
    # Barrido de voltaje: default = TOL_VOLTAJE (ver constantes.py). Se expone
    # por cfg para reproducir corridas viejas (v1.0.0 usaba (1e-12, 1e-8)).
    tol_voltaje = (float(cfg.get("tol_voltage_abs", TOL_VOLTAJE[0])),
                   float(cfg.get("tol_voltage_rel", TOL_VOLTAJE[1])))

    # Iones del electrolito, en el orden de las componentes u_i de U.
    # modo_general=False → binario del paper (sal simétrica): todo como v1.0.0.
    # etiqueta_c → nombres de archivo ("KCl_100.0mM" en binario, igual que v1.0.0).
    iones, modo_general, etiqueta_c = iones_de_cfg(cfg)

    voltajes_pos = np.linspace(0.0,  V_max_V, n_steps)[1:]
    voltajes_neg = np.linspace(0.0, -V_max_V, n_steps)[1:]

    # -------------------------------------------------------------------------
    # 6.0 — JSON del mallador (fuente de verdad de la geometría y los films)
    # -------------------------------------------------------------------------
    json_mallador = f"{INPUT_DIR}/{m_name}_limits.json"
    with open(json_mallador, "r") as f:
        meta_geo = json.load(f)

    tags = meta_geo["tags"]
    films_activos = construir_films_activos(meta_geo, tags)

    # Completar cada film con el tipo/carga elegidos (cfg["films"] manda el
    # ORDEN DE RAMPA; se reordena films_activos para respetarlo)
    asignaciones = {f["side"] if "side" in f else f["name"]: f
                    for f in cfg.get("films", [])}
    for f_dict in films_activos:
        asig = asignaciones.get(f_dict["name"])
        if asig is None:
            raise ValueError(f"La malla tiene film '{f_dict['name']}' pero cfg no "
                             f"trae su carga (clave 'films').")
        f_dict["type"]        = asig["type"]
        f_dict["n_e_per_nm3"] = asig["n_e_per_nm3"]
    orden = [f["side"] if "side" in f else f["name"] for f in cfg.get("films", [])]
    films_activos.sort(key=lambda f: orden.index(f["name"]))

    # -------------------------------------------------------------------------
    # 6.1 — Carga de malla, boundaries y subdominios
    # -------------------------------------------------------------------------
    mesh = Mesh()
    with XDMFFile(f"{INPUT_DIR}/{m_name}_domain.xdmf") as infile:
        infile.read(mesh)

    # Boundaries (1D: facetas/edges)
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    with XDMFFile(f"{INPUT_DIR}/{m_name}_facets.xdmf") as infile:
        infile.read(mvc, "f")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Subdominios (2D: tags de volumen)
    mvc_vol = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(f"{INPUT_DIR}/{m_name}_domain.xdmf") as infile:
        infile.read(mvc_vol, "subdomains")
    subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc_vol)
    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)

    print(f"  [6.1] Mesh loaded: {mesh.num_vertices()} nodes, {mesh.num_cells()} elements")

    # -------------------------------------------------------------------------
    # 6.2 — Tags de carga y derivados físicos
    # -------------------------------------------------------------------------
    # Tags sobre los que se aplica σ (pared + zonas cargadas de la membrana)
    tags_carga = []
    if "WALL" in tags:
        tags_carga.append(tags["WALL"])

    if aplicar_carga_coronas:
        for nombre_tag in ["CHARGE_ZONE_TIP", "CHARGE_ZONE_BASE"]:
            if nombre_tag in tags:
                tags_carga.append(tags[nombre_tag])

    print(f"  [6.2] Charge tags: {tags_carga}")

    # Derivados físicos universales
    epsilon = EPS_0 * eps_r

    print(f"  [6.2] Electrolyte ({'general' if modo_general else 'binary'}, "
          f"{len(iones)} ions): " +
          ", ".join(f"{i['symbol']}(z={int(i['z']):+d}, D={i['D_m2s']:.2e}, c0={i['c0_mM']:g} mM)"
                    for i in iones))

    # -------------------------------------------------------------------------
    # 6.3 — Cálculos físicos de cada film (densidad de carga + Donnan analítico)
    # -------------------------------------------------------------------------
    for f_dict in films_activos:
        # Densidad de carga fija del film
        f_dict["rho_target"] = f_dict["n_e_per_nm3"] * E_CHARGE * 1e27   # [C/m³]
        f_dict["c_fix_eq"]   = f_dict["rho_target"] / F_CONST            # [mol/m³]

        # Equilibrio de Donnan (sin voltaje), como referencia
        #   film positivo → φ_D > 0 ; film negativo → φ_D < 0
        if not modo_general and iones[0]["z"] == 1 and iones[1]["z"] == -1:
            # Sal 1:1: la fórmula analítica del paper, sin cambios
            c_bulk_SI  = c0_mM   # [mM]
            phi_D_anal = (R_GAS * T / F_CONST) * np.arcsinh(f_dict["c_fix_eq"] / (2 * c_bulk_SI))
            f_dict["phi_D_anal_mV"] = phi_D_anal * 1e3

            # Concentraciones analíticas en el film (Donnan):
            #   c_i^film = c_i^bulk * exp(-z_i F φ_D / RT)
            f_dict["cK_film_anal"]  = c_bulk_SI * np.exp(-1    * F_CONST * phi_D_anal / (R_GAS * T))
            f_dict["cCl_film_anal"] = c_bulk_SI * np.exp(-(-1) * F_CONST * phi_D_anal / (R_GAS * T))
        else:
            # z:z con |z| > 1 o modo general: Donnan numérico
            # (Σ z_i·c0_i·exp(−z_i·ψ) + c_fix = 0 dentro del film)
            psi_D = donnan_psi(iones, f_dict["c_fix_eq"])
            f_dict["phi_D_anal_mV"] = psi_D * (R_GAS * T / F_CONST) * 1e3
            c_film = [i["c0_mM"] * np.exp(-i["z"] * psi_D) for i in iones]
            if modo_general:
                f_dict["c_film_anal"] = {i["symbol"]: c for i, c in zip(iones, c_film)}
            else:
                f_dict["cK_film_anal"], f_dict["cCl_film_anal"] = c_film

        print(f"  Film '{f_dict['name']}' ({f_dict['type']}): "
              f"n_e={f_dict['n_e_per_nm3']:.3f} e/nm³ → "
              f"ρ_film={f_dict['rho_target']:.3e} C/m³ | "
              f"Φ_Donnan={f_dict['phi_D_anal_mV']:.2f} mV")

    if not films_activos:
        print("  (no films: no volumetric charge terms are added)")

    # -------------------------------------------------------------------------
    # 6.4 — Formulación variacional (sistema PNP, con término de film opcional)
    # -------------------------------------------------------------------------
    # Espacio mixto: φ + una variable logarítmica u_i por ion → (1 + N) campos.
    # Con 2 iones es el P1×P1×P1 (φ, up, um) del paper.
    P1 = FiniteElement('P', mesh.ufl_cell(), GRADO_ELEMENTO)
    V  = FunctionSpace(mesh, MixedElement([P1] * (1 + len(iones))))

    U          = Function(V)
    phi, *us   = split(U)
    v_phi, *vs = TestFunctions(V)

    # Coordenada radial para integración axisimétrica (factor r)
    x_coord = SpatialCoordinate(mesh)
    r       = x_coord[1]

    # Diagnóstico: verificar que cada film existe en la malla (volumen > 0)
    if films_activos:
        print(f"\n  [DIAGNOSTIC] Volumes:")
        vol_fluid = assemble(1.0 * r * dx(tags["DOMAIN_FLUID"]))
        print(f"    DOMAIN_FLUID (tag {tags['DOMAIN_FLUID']}): {vol_fluid:.3e}")
        for f_dict in films_activos:
            vol = assemble(1.0 * r * dx(f_dict["tag_domain"]))
            print(f"    DOMAIN_FILM_{f_dict['name'].upper()} "
                  f"(tag {f_dict['tag_domain']}): {vol:.3e}  <-- Must be > 0")
            if vol <= 0:
                print(f"  [WARNING] Film '{f_dict['name']}' not detected in the mesh!")

    # Concentraciones: variables logarítmicas → concentración lineal
    #   c_i = c0_i · exp(u_i)   (binario: cp = c0·exp(up), cm = c0·exp(um))
    cs = [ion["c0_mM"] * exp(u) for ion, u in zip(iones, us)]

    sigma_val = Constant(0.0)   # se rampa en la ETAPA 1

    # Ecuación de Poisson
    # ρ = F · Σ z_i c_i, acumulado término a término: con 2 iones es la misma
    # expresión F·(z_p·cp + z_m·cm) del paper.
    f_pois = (epsilon * R_GAS * T) / F_CONST
    suma_zc = iones[0]["z"] * cs[0]
    for ion, c_i in zip(iones[1:], cs[1:]):
        suma_zc = suma_zc + ion["z"] * c_i
    rho    = F_CONST * suma_zc

    carga_ds = ds(tags_carga[0])
    for t in tags_carga[1:]:
        carga_ds += ds(t)

    F_pois = (  f_pois * dot(grad(phi), grad(v_phi)) * r * dx
              - rho * v_phi * r * dx
              - sigma_val * v_phi * r * carga_ds )

    # Un término de carga volumétrica por cada film activo. El Constant rho_fix
    # de cada film (hoy 0) se rampa en la ETAPA 1b. Sin films, este bucle no
    # agrega nada → Poisson queda en su forma "sin film".
    for f_dict in films_activos:
        F_pois += - f_dict["rho_fix"] * v_phi * r * dx(f_dict["tag_domain"])

    # Ecuaciones de Nernst-Planck, una por ion:  J_i = −D_i c_i (∇u_i + z_i ∇φ)
    Js = [-ion["D_m2s"] * c_i * (grad(u) + ion["z"] * grad(phi))
          for ion, c_i, u in zip(iones, cs, us)]

    suma_np = dot(Js[0], grad(vs[0]))
    for J_i, v_i in zip(Js[1:], vs[1:]):
        suma_np = suma_np + dot(J_i, grad(v_i))
    F_np    = suma_np * r * dx
    F_total = F_pois + F_np

    print(f"  [6.4] PNP formulation assembled "
          f"({1 + len(iones)} fields: φ + {len(iones)} ions | "
          f"films: {len(films_activos)} → {[f['name'] for f in films_activos]})")

    # -------------------------------------------------------------------------
    # 6.5 — Condiciones de borde y configuración del solver Newton
    # -------------------------------------------------------------------------
    # Reservorios: φ = V (INLET) / 0 (OUTLET) y u_i = 0 (c_i = c0_i) en ambos.
    # Mismo orden que el paper: INLET (φ, u_1..u_N), después OUTLET.
    bc_phi_in = Expression("val", val=0.0, degree=0)

    bcs = []
    for nombre_tag, bc_phi in (("INLET", bc_phi_in), ("OUTLET", Constant(0.0))):
        bcs.append(DirichletBC(V.sub(0), bc_phi, boundaries, tags[nombre_tag]))
        for k in range(1, len(iones) + 1):
            bcs.append(DirichletBC(V.sub(k), Constant(0.0), boundaries, tags[nombre_tag]))

    problema   = NonlinearVariationalProblem(F_total, U, bcs, derivative(F_total, U))
    solver_nls = NonlinearVariationalSolver(problema)
    solver_nls.parameters['newton_solver']['linear_solver']      = LINEAR_SOLVER
    solver_nls.parameters['newton_solver']['maximum_iterations'] = NEWTON_MAX_ITER
    solver_nls.parameters['newton_solver']['report']             = NEWTON_REPORT

    print(f"  [6.5] Newton solver configured with {LINEAR_SOLVER.upper()}")

    # -------------------------------------------------------------------------
    # 6.6 — Post-procesamiento: corriente / avance adaptativo de voltaje
    # -------------------------------------------------------------------------
    def calcular_corriente(sol):
        """Calcula I_in e I_out en nA. Factor 2π por simetría axial."""
        s_phi, *s_us = split(sol)
        suma_zJ = None
        for ion, s_u in zip(iones, s_us):
            s_c   = ion["c0_mM"] * exp(s_u)
            J_sol = -ion["D_m2s"] * s_c * (grad(s_u) + ion["z"] * grad(s_phi))
            suma_zJ = ion["z"] * J_sol if suma_zJ is None else suma_zJ + ion["z"] * J_sol
        J_e     = F_CONST * suma_zJ
        n       = FacetNormal(mesh)
        I_in  = assemble(dot(J_e, n) * FACTOR_AXISIM * r * ds(tags["INLET"]))  * A_TO_NA
        I_out = assemble(dot(J_e, n) * FACTOR_AXISIM * r * ds(tags["OUTLET"])) * A_TO_NA
        return I_in, I_out

    # (V, iteraciones de Newton) de cada solve del barrido de voltaje, incluidos
    # los pasos intermedios de la subdivisión. Lo usa el sanity check (3).
    iteraciones_voltaje = []

    def avanzar_voltaje(v_objetivo, v_desde, max_subdivisiones=MAX_SUBDIVISIONES):
        """
        Avanza de v_desde a v_objetivo subdividiendo el paso si Newton diverge.
        Solo modifica U y bc_phi_in (y rehace solver_nls tras un fallo). No
        guarda nada.
        """
        nonlocal solver_nls
        pasos = [v_objetivo]
        v_actual = v_desde

        while pasos:
            v_next = pasos[-1]
            U_backup = U.copy(deepcopy=True)
            bc_phi_in.val = v_next * F_CONST / (R_GAS * T)

            try:
                n_iter, _convergio = solver_nls.solve()
                iteraciones_voltaje.append((v_next, int(n_iter)))
                v_actual = v_next
                pasos.pop()
            except RuntimeError:
                # Un paso que diverge deja NaN en el LU de MUMPS, y PETSc recuerda
                # el fallo del precondicionador (DIVERGED_PC_FAILED): con el MISMO
                # solver todo reintento fallaba en la iteración 0, aun con el paso
                # más chico (1/64 del original), así que la subdivisión no
                # rescataba nada y el barrido se cortaba en ese voltaje. Se usa
                # un solver nuevo con los mismos parámetros.
                nuevo = NonlinearVariationalSolver(problema)
                nuevo.parameters.update(solver_nls.parameters)
                solver_nls = nuevo
                U.assign(U_backup)
                bc_phi_in.val = v_actual * F_CONST / (R_GAS * T)
                v_medio = (v_actual + v_next) / 2
                if len(pasos) > max_subdivisiones or abs(v_medio - v_actual) < 1e-4:
                    raise RuntimeError(f"Does not converge at V={v_next:+.4f}V even with subdivision")
                print(f"        [adaptive] failure at {v_next:+.4f}V → intermediate step {v_medio:+.4f}V")
                pasos.append(v_medio)

    # -------------------------------------------------------------------------
    # 6.7 — Ejecución: 3 etapas (rampa σ → rampa ρ_film → barrido de voltaje)
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    archivo_h5   = os.path.join(OUTPUT_DIR, f"Solutions_{m_name}_{etiqueta_c}.h5")
    archivo_txt  = os.path.join(OUTPUT_DIR, f"IV_curve_{m_name}_{etiqueta_c}.txt")
    archivo_json = os.path.join(OUTPUT_DIR, f"Solutions_{m_name}_{etiqueta_c}_sim.json")

    hdf = HDF5File(mesh.mpi_comm(), archivo_h5, "w")
    hdf.write(mesh, "/malla")

    print("\n" + "="*60)
    print("  >>> STARTING SIMULATION (3 STAGES) <<<")
    print("="*60)

    # -------------------------------------------------------------------------
    # CHECKPOINT DE EQUILIBRIO — las Etapas 1+1b son ~85 factorizaciones de
    # MUMPS que dan SIEMPRE el mismo resultado para los mismos parámetros
    # (verificado dígito a dígito entre corridas). Se guardan en
    # RESULTS/equilibria/ junto a un .json con TODOS los parámetros que
    # definen el equilibrio; solo se reutiliza si coinciden EXACTO.
    # Las difusividades NO entran en la clave: en equilibrio el flujo es cero y
    # la solución no depende de D (sí de las valencias z).
    # Modo general: la clave lleva la lista de iones (z, c0) EN ORDEN, porque el
    # orden fija qué componente de U es cada ion. Modo binario: la clave de
    # v1.0.0, sin cambios (los checkpoints existentes siguen sirviendo).
    # -------------------------------------------------------------------------
    if not modo_general:
        params_eq = {
            "m_name": m_name, "input_dir": INPUT_DIR,
            "n_nodos": int(mesh.num_vertices()), "n_celdas": int(mesh.num_cells()),
            "c0_mM": float(c0_mM), "sigma_Cm2": float(sigma_Cm2),
            "rings": bool(aplicar_carga_coronas),
            "T": float(T), "eps_r": float(eps_r),
            "z_p": float(iones[0]["z"]), "z_m": float(iones[1]["z"]),
            "films": [{"name": f["name"], "type": str(f["type"]),
                       "rho_target": float(f["rho_target"])} for f in films_activos],
        }
    else:
        params_eq = {
            "m_name": m_name, "input_dir": INPUT_DIR,
            "n_nodos": int(mesh.num_vertices()), "n_celdas": int(mesh.num_cells()),
            "ions": [{"z": float(i["z"]), "c0_mM": float(i["c0_mM"])} for i in iones],
            "sigma_Cm2": float(sigma_Cm2),
            "rings": bool(aplicar_carga_coronas),
            "T": float(T), "eps_r": float(eps_r),
            "films": [{"name": f["name"], "type": str(f["type"]),
                       "rho_target": float(f["rho_target"])} for f in films_activos],
        }

    def _buscar_checkpoint_eq():
        """Busca en EQ_DIR un checkpoint cuyo .json coincida EXACTO con params_eq."""
        if not os.path.isdir(EQ_DIR):
            return None
        objetivo = json.loads(json.dumps(params_eq))   # normalizar tipos como JSON
        for fn in sorted(os.listdir(EQ_DIR)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(EQ_DIR, fn)) as f:
                    meta = json.load(f)
            except Exception:
                continue
            h5p = os.path.join(EQ_DIR, fn[:-5] + ".h5")
            if meta == objetivo and os.path.exists(h5p):
                return h5p
        return None

    def _guardar_checkpoint_eq(U_eq):
        """Guarda U_eq + params_eq en EQ_DIR (nombre legible + hash corto)."""
        os.makedirs(EQ_DIR, exist_ok=True)
        h8 = hashlib.md5(json.dumps(params_eq, sort_keys=True).encode()).hexdigest()[:8]
        base = os.path.join(EQ_DIR, f"eq_{m_name}_{etiqueta_c}_{h8}")
        h5out = HDF5File(mesh.mpi_comm(), base + ".h5", "w")
        h5out.write(U_eq, "/U_zero")
        h5out.close()
        with open(base + ".json", "w") as f:
            json.dump(params_eq, f, indent=2)
        print(f"  [CHECKPOINT] Equilibrium saved: {base}.h5")

    # Política de reutilización del checkpoint:
    #   "ask"       → como el solver original (pregunta si encuentra uno)
    #   "auto"/True → usarlo sin preguntar si existe uno compatible
    #   False       → recalcular siempre (ignora checkpoints existentes)
    U_zero  = None
    ck_path = None
    if reusar_equilibrio is not False:
        ck_path = _buscar_checkpoint_eq()
    if ck_path:
        print(f"\n  [CHECKPOINT] Compatible equilibrium found:\n      {ck_path}")
        print("                (Note: the V=0 equilibrium depends on the valences and charges,")
        print("                not on the salt name or its diffusivities. It is physically")
        print("                correct to reuse it between salts of the same type, e.g. KCl and KClO4).")
        usar = True if reusar_equilibrio in ("auto", True) else \
               _preguntar_ok("  Load it and skip Stages 1/1b?")
        if usar:
            h5in = HDF5File(mesh.mpi_comm(), ck_path, "r")
            h5in.read(U, "/U_zero")
            h5in.close()
            # CRÍTICO: dejar las Constants en sus valores finales (el checkpoint
            # guarda U, no el estado de sigma_val/rho_fix).
            sigma_val.assign(sigma_Cm2)
            for f_dict in films_activos:
                f_dict["rho_fix"].assign(f_dict["rho_target"])
            # Verificación: el residual del estado cargado ya debe estar bajo la
            # tolerancia estricta → este solve debe terminar en 0 iteraciones.
            solver_nls.parameters['newton_solver']['absolute_tolerance'] = TOL_EQUILIBRIO[0]
            solver_nls.parameters['newton_solver']['relative_tolerance'] = TOL_EQUILIBRIO[1]
            solver_nls.solve()
            U_zero  = U.copy(deepcopy=True)
            phi_arr = U_zero.split(True)[0].compute_vertex_values(mesh)
            print(f"  [CHECKPOINT ✓] Equilibrium loaded and verified: "
                  f"φ min={np.min(phi_arr):.4f}, max={np.max(phi_arr):.4f}")

    if U_zero is None:
        # ETAPA 1: rampa de carga superficial σ con V = 0 (tolerancias estrictas)
        n_steps_carga = n_steps_sigma
        pasos_sigma   = np.linspace(0, sigma_Cm2, n_steps_carga)

        print(f"\n  [STAGE 1] σ ramp: 0 → {sigma_Cm2} C/m² ({n_steps_carga} steps)")
        rampa_adaptativa(solver_nls, sigma_val, pasos_sigma,
                         tol_sigma[0], tol_sigma[1], tag_desc="Ramp-σ")

        U_zero  = U.copy(deepcopy=True)
        phi_arr = U_zero.split(True)[0].compute_vertex_values(mesh)
        print(f"\n  [STAGE 1 ✓] φ equilibrium: min={np.min(phi_arr):.4f}, max={np.max(phi_arr):.4f}")

        # ETAPA 1b: rampa de carga volumétrica ρ_film, SECUENCIAL por cada film
        # activo. Se cargan en el orden de films_activos (= orden de cfg["films"]).
        # Secuencial = primero un film completo, después el siguiente: si falla,
        # queda claro CUÁL film y en qué paso de su rampa se rompió la
        # convergencia. Sin films, este bloque no hace nada.
        if films_activos:
            print(f"\n  [STAGE 1b] ρ_film ramp (sequential, "
                  f"order: {', '.join(f['name'] for f in films_activos)})")
            print(f"  [CRITICAL] STRICT tolerances: abs={tol_film[0]}, rel={tol_film[1]}")

            for f_dict in films_activos:
                pasos_rho_film = np.linspace(0, f_dict["rho_target"], n_steps_film)
                print(f"\n    Film '{f_dict['name']}': 0 → {f_dict['rho_target']:.3e} C/m³ "
                      f"({n_steps_film} steps)")
                rampa_adaptativa(solver_nls, f_dict["rho_fix"], pasos_rho_film,
                                 tol_film[0], tol_film[1],
                                 tag_desc=f"Ramp-ρ_film[{f_dict['name']}]")

            U_zero  = U.copy(deepcopy=True)
            phi_arr = U_zero.split(True)[0].compute_vertex_values(mesh)
            print(f"\n  [STAGE 1b ✓] φ with film(s): min={np.min(phi_arr):.4f}, max={np.max(phi_arr):.4f}")

        # Guardar el equilibrio recién calculado para las próximas corridas
        _guardar_checkpoint_eq(U_zero)

    # Guardar solución a 0 V (con σ y film, si aplica)
    hdf.write(U_zero, "/U_+0.00V")

    # ETAPA 2: barrido de voltaje adaptativo (ver TOL_VOLTAJE en constantes.py)
    solver_nls.parameters['newton_solver']['absolute_tolerance'] = tol_voltaje[0]
    solver_nls.parameters['newton_solver']['relative_tolerance'] = tol_voltaje[1]
    bc_phi_in.val = 0.0

    resultados_IV = []
    I_in_0, I_out_0 = calcular_corriente(U_zero)
    resultados_IV.append((0.0, I_in_0, I_out_0))
    if "callback_iv" in cfg and callable(cfg["callback_iv"]):
        try:
            cfg["callback_iv"](list(resultados_IV))
        except Exception:
            pass

    print(f"\n  [STAGE 2] Adaptive voltage sweep (abs_tol={tol_voltaje[0]}, rel_tol={tol_voltaje[1]})")

    print(f"\n    Positive branch (0V → +{V_max_V}V):")
    U.assign(U_zero)
    v_actual = 0.0
    for v in voltajes_pos:
        try:
            avanzar_voltaje(v, v_actual)
            v_actual = v
            I_in, I_out = calcular_corriente(U)
            resultados_IV.append((v, I_in, I_out))
            if guardar_todas_sol or abs(v / 0.1 - round(v / 0.1)) < 1e-6:
                hdf.write(U, f"/U_{v:+.2f}V")
            print(f"      V={v:+.2f}V  I_in={-I_in:+.4f} nA  I_out={I_out:+.4f} nA")
            if "callback_iv" in cfg and callable(cfg["callback_iv"]):
                try:
                    cfg["callback_iv"](list(resultados_IV))
                except Exception:
                    pass
        except RuntimeError as e:
            print(f"      [FAILED] V={v:+.2f}V: {str(e)[:80]}")
            break

    print(f"\n    Negative branch (0V → -{V_max_V}V):")
    U.assign(U_zero)
    v_actual = 0.0
    for v in voltajes_neg:
        try:
            avanzar_voltaje(v, v_actual)
            v_actual = v
            I_in, I_out = calcular_corriente(U)
            resultados_IV.append((v, I_in, I_out))
            if guardar_todas_sol or abs(v / 0.1 - round(v / 0.1)) < 1e-6:
                hdf.write(U, f"/U_{v:+.2f}V")
            print(f"      V={v:+.2f}V  I_in={-I_in:+.4f} nA  I_out={I_out:+.4f} nA")
            if "callback_iv" in cfg and callable(cfg["callback_iv"]):
                try:
                    cfg["callback_iv"](list(resultados_IV))
                except Exception:
                    pass
        except RuntimeError as e:
            print(f"      [FAILED] V={v:+.2f}V: {str(e)[:80]}")
            break

    hdf.close()

    # -------------------------------------------------------------------------
    # 6.8 — Chequeos físicos / numéricos (sanity checks)
    #
    # El solver es PNP estacionario SIN física que limite la corriente (no hay
    # precipitación ni reacción dentro del motor). Por lo tanto, para una
    # geometría fija: (a) I_in debe balancear a I_out, y (b) |I| debe crecer con
    # |V|. Una violación apunta a la NUMÉRICA, no al modelo:
    #   - desbalance in/out  -> la malla prehecha es demasiado gruesa para la
    #     concentración / carga de film elegida (capa de Debye sub-resuelta).
    #   - |I| que baja con |V| -> Newton cayó en una rama no convergida;
    #     achicar el paso de V o subir los steps de carga suele arreglarlo.
    # Cada aviso es un dict {level, code, v, msg}: una sola fuente de verdad que
    # cada front-end (GUI, consola, JSON de batch) renderiza por severidad.
    # -------------------------------------------------------------------------
    resultados_IV.sort(key=lambda x: x[0])
    warnings_list = []
    FLOOR_I  = 1e-6   # nA: por debajo de esto la corriente es ~cero numérico (V≈0)
    UMBRAL_PCT = 5.0  # % a partir del cual se emite el aviso (ambos chequeos)

    # (1) Conservación de corriente: |I_in + I_out| relativo a la magnitud media.
    #     Signos: n saliente en ambas tapas -> conservación es I_in + I_out = 0,
    #     o sea corriente física entrante = -I_in y saliente = I_out.
    for v, i_in, i_out in resultados_IV:
        i_in_fis, i_out_fis = -i_in, i_out
        mag_media = (abs(i_in_fis) + abs(i_out_fis)) / 2.0
        if mag_media > FLOOR_I:
            err_pct = abs(i_in_fis - i_out_fis) / mag_media * 100
            if err_pct > UMBRAL_PCT:
                warnings_list.append({
                    "level": "warning",
                    "code":  "current_conservation",
                    "v":     v,
                    "msg":  (f"Inlet/outlet current mismatch of {err_pct:.1f}% at "
                             f"V={v:+.2f}V. PoroSIM's pre-built mesh is too coarse "
                             f"for this electrolyte concentration and/or applied "
                             f"film/surface charge (thin Debye layers need finer "
                             f"near-wall resolution), so this point is unreliable. "
                             f"Future PoroSIM versions will add selectable mesh types."),
                })

    # (2) |I| debe crecer con |V|. Se compara una sola corriente (la física de
    #     entrada, -I_in) para no mezclar el desbalance in/out que ya vigila (1).
    rama_pos = sorted([r for r in resultados_IV if r[0] >= 0.0], key=lambda x: x[0])
    rama_neg = sorted([r for r in resultados_IV if r[0] <= 0.0], key=lambda x: x[0], reverse=True)

    def _chequear_monotonia(rama, nombre_rama):
        for k in range(1, len(rama)):
            v_curr, iin_c, _ = rama[k]
            v_prev, iin_p, _ = rama[k - 1]
            i_curr = abs(-iin_c)
            i_prev = abs(-iin_p)
            if i_prev > FLOOR_I and i_curr < i_prev:
                caida_pct = (i_prev - i_curr) / i_prev * 100
                if caida_pct > UMBRAL_PCT:
                    warnings_list.append({
                        "level": "warning",
                        "code":  "non_monotonic_current",
                        "v":     v_curr,
                        "msg":  (f"Current magnitude drops by {caida_pct:.1f}% from "
                                 f"V={v_prev:+.2f}V to V={v_curr:+.2f}V ({nombre_rama} "
                                 f"branch). In this model current should grow with "
                                 f"|V|, so this is physically incorrect and usually "
                                 f"means the solver did not fully converge. Try "
                                 f"smaller voltage steps, or increase the "
                                 f"surface-charge / film ramp steps."),
                    })

    _chequear_monotonia(rama_pos, "positive")
    _chequear_monotonia(rama_neg, "negative")

    # (3) Pasos de voltaje que convergieron en menos de NEWTON_MIN_ITER_VOLTAJE
    #     iteraciones: es la firma de una tolerancia inefectiva (Newton devuelve
    #     el paso linealizado como si fuera la solución). Ni (1) ni (2) lo
    #     detectan: el desbalance in/out queda < 1 % aun sin converger.
    pasos_cortos = [(v, k) for v, k in iteraciones_voltaje
                    if k < NEWTON_MIN_ITER_VOLTAJE]
    if pasos_cortos:
        lista_v = ", ".join(f"{v:+.2f}V" for v, _ in pasos_cortos[:8])
        if len(pasos_cortos) > 8:
            lista_v += ", ..."
        warnings_list.append({
            "level": "warning",
            "code":  "newton_few_iterations",
            "v":     pasos_cortos[0][0],
            "msg":  (f"{len(pasos_cortos)} of {len(iteraciones_voltaje)} voltage "
                     f"steps converged in fewer than {NEWTON_MIN_ITER_VOLTAJE} "
                     f"Newton iterations ({lista_v}). With an ineffective "
                     f"tolerance Newton returns the linearised step as if it "
                     f"were the solution, and the I-V then depends on the "
                     f"voltage step. Check tol_voltage_abs / tol_voltage_rel "
                     f"(defaults in constantes.TOL_VOLTAJE)."),
        })

    if warnings_list:
        print("\n" + "!" * 64)
        print("  >>> SANITY-CHECK WARNINGS <<<")
        for w in warnings_list:
            print(f"  [{w['level'].upper()}] {w['msg']}")
        print("!" * 64)

    # -------------------------------------------------------------------------
    # 6.9 — Exportar tabla I-V
    # -------------------------------------------------------------------------
    with open(archivo_txt, "w") as f:
        f.write("Voltage(V)\tI_in(nA)\tI_out(nA)\n")
        for v, i_in, i_out in resultados_IV:
            f.write(f"{v:+.2f}\t{-i_in:.6f}\t{+i_out:.6f}\n")

    # -------------------------------------------------------------------------
    # 6.10 — Exportar JSON de simulación (fusión geometría + simulación)
    # -------------------------------------------------------------------------
    sim_json = {
        **meta_geo,
        "m_name": m_name,
        "warnings": warnings_list,

        "simulation": {
            # Bloque coherente de la sal (del catálogo o custom): toda la
            # info de la sal viaja junta y los módulos la leen como una unidad.
            # Modo general: "c0_mM" = null a propósito (no hay UNA c0 por ion):
            # un lector binario falla en vez de usar una c0 equivocada.
            "salt":          sal,
            "c0_mM":        None if modo_general else c0_mM,
            "c0_inlet_mM":  None if modo_general else c0_mM,
            "c0_outlet_mM": None if modo_general else c0_mM,
            "sigma_Cm2":    sigma_Cm2,
            "T_K":          T,
            "eps_r":        eps_r,
            "V_max_V":      V_max_V,
            "n_steps":      n_steps,
            "charge_tags":   tags_carga,
            # Tolerancia de Newton del barrido (trazabilidad: v1.0.0 usaba
            # (1e-12, 1e-8), que no convergía los pasos de voltaje).
            "tol_voltage_abs": tol_voltaje[0],
            "tol_voltage_rel": tol_voltaje[1],
            # Iones en el ORDEN de las componentes de U = (φ, u_1, ..., u_N):
            # es lo que necesita quien lea el .h5 para armar el espacio mixto
            # con 1 + N campos (binario: N = 2, el P1×P1×P1 de siempre).
            "ions":          iones,
        }
    }
    if modo_general:
        sim_json["simulation"]["electrolyte"] = (
            cfg.get("electrolyte") or
            ([{"salt": sal["name"], "c0_mM": c0_mM}] if sal else None))
        sim_json["simulation"]["electrolyte_label"] = etiqueta_c

    # Lista de films activos (vacía si no hay films). Cada entrada lleva su
    # lado, tipo y parámetros físicos derivados. El orden refleja el orden de
    # rampa.
    def _film_json(f_dict):
        d = {
            "side":                f_dict["name"],
            "type":                f_dict["type"],
            "n_e_per_nm3":         f_dict["n_e_per_nm3"],
            "rho_fix_target_Cm3":  f_dict["rho_target"],
            "c_fix_eq_molm3":      f_dict["c_fix_eq"],
            "phi_D_anal_mV":       f_dict["phi_D_anal_mV"],
        }
        if modo_general:
            d["c_film_anal_molm3"]   = f_dict["c_film_anal"]
        else:
            d["cK_film_anal_molm3"]  = f_dict["cK_film_anal"]
            d["cCl_film_anal_molm3"] = f_dict["cCl_film_anal"]
        return d

    sim_json["simulation"]["films"] = [_film_json(f_dict) for f_dict in films_activos]

    with open(archivo_json, "w") as f:
        json.dump(sim_json, f, indent=2)

    print("\n" + "="*60)
    print("  >>> SIMULATION COMPLETED <<<")
    print("="*60)
    print(f"  HDF5 solutions  : {archivo_h5}")
    print(f"  JSON metadata   : {archivo_json}")
    print(f"  I-V table       : {archivo_txt}")
    print("="*60)

    return {
        "archivo_h5":   archivo_h5,
        "archivo_txt":  archivo_txt,
        "archivo_json": archivo_json,
        "IV":           resultados_IV,
        "warnings":     warnings_list,
    }
