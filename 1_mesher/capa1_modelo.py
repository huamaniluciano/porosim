# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
  POROSIM — MALLADOR · CAPA 1: MODELO (topología declarativa: puntos + líneas)
  SIN Gmsh todavía. Solo deriva la topología y la dibuja en matplotlib
  para validarla a ojo en los 4 casos (sin film / tip / base / ambos).
═══════════════════════════════════════════════════════════════════════════

  QUÉ HACE ESTA CAPA
  ──────────────────
  A partir de los ticks (include_film_tip / include_film_base) y los
  parámetros geométricos, DERIVA:
    • las estaciones axiales (z) y los slabs entre ellas,
    • los puntos en cada estación (niveles radiales),
    • las LÍNEAS que existen, aplicando UNA sola regla:
          existe línea en una banda  ⟺  región_izq ≠ región_der
    • el rol, el tag físico y la DIRECCIÓN de cada línea.

  Los nombres lógicos de las líneas ESPEJAN el dibujador tip actual
  (l_inlet, l_tip_charge, l_axis_pore, l_film_tip_face, ...), de modo que
  para la config "film tip" el manifiesto coincide 1 a 1 con el dibujador
  validado. Para las configs nuevas (base / ambos / ninguno), los nombres
  se derivan por la misma regla de forma simétrica.

  QUÉ NO HACE (todavía)
  ─────────────────────
  • NO arma curve loops ni superficies orientadas  → Capa 2.
  • NO emite a Gmsh                                 → Capa 2.
  • NO refina ni mallea                             → mesher.

  El objetivo de la Capa 1 es que el CONJUNTO de líneas y sus nombres sean
  correctos antes de meterse con las orientaciones de los loops (Capa 2).

  Migración futura a derivación automática (opción A): el modelo de slabs y
  la regla de existencia ya son declarativos; solo el ENSAMBLADO de loops
  (Capa 2) usará plantillas. Cambiar a derivación pura tocaría únicamente esa
  parte, sin reescribir este modelo.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# REGIONES Y CONTRATO DE TAGS
# =================================================================
FLUID = "FLUID"
FILM  = "FILM"
VOID  = "VOID"      # membrana sólida: NO se malla
EXT   = "EXT"       # exterior del dominio

TAGS = {
    "AXIS":                1,
    "WALL":                2,
    "INLET":               3,
    "OUTLET":              4,
    "CHARGE_ZONE_TIP":     5,
    "CHARGE_ZONE_BASE":    6,
    "FILM_TIP_INTERFACE":  7,
    "FILM_BASE_INTERFACE": 8,
    "DOMAIN_FLUID":        10,
    "DOMAIN_FILM_TIP":     11,
    "DOMAIN_FILM_BASE":    12,
}

NM = 1e9


# =================================================================
# PARÁMETROS
# =================================================================
@dataclass
class Params:
    L_pore: float = 50e-9
    D_tip:  float = 10e-9
    D_base: float = 50e-9
    L_res:  float = 500e-9
    R_res:  float = 400e-9
    L_charge: float = 5e-9
    L_far:    float = 45e-9
    # Films: por DEFAULT no hay ninguno (principio de menor sorpresa para el
    # geom.json: un JSON mínimo produce un canal pelado; los films se piden
    # explícitamente). Todos los usos internos pasan estos flags explícitos.
    include_film_tip:  bool  = False
    delta_film_tip:    float = 10e-9
    include_film_base: bool  = False
    delta_film_base:   float = 10e-9
    # Puntos intermedios para la spline de la pared del canal (Capa 3).
    # Para paredes curvas (bullet), muchos puntos evitan que el spline de Gmsh
    # se vea recta. 20 es muy poco; 200 es un buen balance (se crea una sola vez).
    #
    # TODO(genericidad): este default es FIJO y no escala con la geometría — un
    # bullet de 300 nm y uno de 12 µm reciben los mismos 200 puntos (y la GUI
    # ni lo expone). Contrastar con capa4, donde el Sampling del campo Distance
    # SÍ se auto-calcula (max(400, L_pore/0.2nm)). Pendiente: derivarlo de la
    # geometría (p.ej. en función de L_pore/h_param, que fijan la curvatura del
    # bullet) en vez de un número mágico. Mientras tanto: para bullets muy
    # largos con h chico, subirlo a mano desde el geom.json si la spline se ve
    # facetada en el tip al inspeccionar la malla.
    N_PTS_WALL: int = 200

    # Perfil de la pared del canal. Despacha en R_canal():
    #   "cylinder" / "conical" → lineal tip→base (cilindro si R_tip==R_base).
    #   "bullet"              → exponencial R(x) = R_base - (R_base-R_tip)·exp(-x/h_param),
    #                           con x = z - z_tip (distancia desde la boca tip).
    channel_type: str = "conical"
    h_param: float = 50e-9   # escala de transición del bullet [m] (ignorado si no es bullet)

    @property
    def R_tip(self):  return self.D_tip / 2.0
    @property
    def R_base(self): return self.D_base / 2.0
    @property
    def z_inlet(self):     return 0.0
    @property
    def z_tip(self):       return self.L_res
    @property
    def z_base(self):      return self.L_res + self.L_pore
    @property
    def z_outlet(self):    return 2.0 * self.L_res + self.L_pore
    @property
    def z_film_tip(self):  return self.z_tip - self.delta_film_tip
    @property
    def z_film_base(self): return self.z_base + self.delta_film_base


# =================================================================
# MODELO: ESTACIONES Y SLABS
# =================================================================
@dataclass
class Station:
    name: str
    z: float

@dataclass
class Slab:
    name: str           # CANAL / FILM_TIP / FILM_BASE / RESERVORIO
    z0: float
    z1: float
    region: str         # FLUID o FILM (región base meshada)
    izq: Station
    der: Station


def construir_estaciones(p: Params):
    st = [Station("INLET", p.z_inlet)]
    if p.include_film_tip:
        st.append(Station("FILM_TIP", p.z_film_tip))
    st.append(Station("TIP", p.z_tip))
    st.append(Station("BASE", p.z_base))
    if p.include_film_base:
        st.append(Station("FILM_BASE", p.z_film_base))
    st.append(Station("OUTLET", p.z_outlet))
    return st


def _slab_kind(a: Station, b: Station):
    if a.name == "TIP" and b.name == "BASE":
        return "CANAL"
    if a.name == "FILM_TIP" and b.name == "TIP":
        return "FILM_TIP"
    if a.name == "BASE" and b.name == "FILM_BASE":
        return "FILM_BASE"
    return "RESERVORIO"


def _slab_region(kind):
    return FILM if kind in ("FILM_TIP", "FILM_BASE") else FLUID


def construir_slabs(estaciones, p: Params):
    slabs = []
    for a, b in zip(estaciones[:-1], estaciones[1:]):
        kind = _slab_kind(a, b)
        slabs.append(Slab(kind, a.z, b.z, _slab_region(kind), a, b))
    return slabs


def slab_suffix(slab: Slab):
    """Sufijo de naming del slab (para l_axis_<suf>, l_top_<suf>)."""
    if slab.name == "CANAL":     return "pore"
    if slab.name == "FILM_TIP":  return "film_tip"
    if slab.name == "FILM_BASE": return "film_base"
    # RESERVORIO: lado tip o lado base según la estación de borde
    if slab.izq.name == "INLET":  return "res_tip"
    if slab.der.name == "OUTLET": return "res_base"
    return "res"   # no debería ocurrir


# =================================================================
# GEOMETRÍA: PARED, REGIONES, NIVELES RADIALES
# =================================================================
def perfil_radio(tipo, R_tip, R_base, z, z_tip, L_pore, h_param=None):
    """FÓRMULA PURA del perfil de la pared (sin Params). FUENTE ÚNICA DE
    VERDAD: la usan R_canal (capas) y el dibujo de la GUI (gui_dibujo), así
    que el tipo de canal se decide acá y en ningún otro lado.

      - "cylinder"/"conical": lineal entre tip y base (cilindro si R_tip==R_base).
      - "bullet": exponencial R(x) = R_base - (R_base - R_tip)·exp(-x/h_param),
        con x = z - z_tip. En x=0 da R_tip; tiende a R_base al alejarse de la
        boca tip (la boca base queda anclada en R_base por el spline de Capa 3).
        Si h_param no está definido, cae al perfil lineal (guarda de la GUI).

    OJO: la Capa 4 lleva esta MISMA fórmula en sintaxis MathEval de Gmsh
    (string, no llamable); si se cambia acá hay que actualizarla allá."""
    if tipo == "bullet" and h_param:
        x = z - z_tip
        return R_base - (R_base - R_tip) * np.exp(-x / h_param)
    return R_tip + (R_base - R_tip) * (z - z_tip) / L_pore


def R_canal(p: Params, z):
    """Radio de la pared del canal a la altura z, desde un Params. Wrapper de
    perfil_radio() (la fórmula única).

    NOTA: se considera FLUID estrictamente para r < R_canal; en r == R_canal
    (la pared) la región es VOID. Los breakpoints se evalúan en el punto medio
    de cada banda, que nunca cae exactamente en la pared, así que el criterio
    estricto no genera ambigüedad."""
    return perfil_radio(getattr(p, "channel_type", "conical"),
                        p.R_tip, p.R_base, z, p.z_tip, p.L_pore,
                        getattr(p, "h_param", None))


def region_de(slab, r, z, p: Params):
    """Región (FLUID/FILM/VOID/EXT) del slab en (r, z)."""
    if slab is None:
        return EXT
    if slab.name == "CANAL":
        return FLUID if r < R_canal(p, z) else VOID
    return slab.region


def niveles_de(station: Station, p: Params):
    """Lista [(nombre_nivel, r)] de la cara vertical en esa estación,
    de adentro (eje) hacia afuera (techo)."""
    if station.name in ("TIP", "BASE"):
        m = p.R_tip if station.name == "TIP" else p.R_base
        return [("axis",   0.0),
                ("mouth",  m),
                ("charge", m + p.L_charge),
                ("far",    m + p.L_charge + p.L_far),
                ("top",    p.R_res)]
    return [("axis", 0.0), ("top", p.R_res)]


def _skey(station: Station):
    """Clave de naming de la estación: 'FILM_TIP' → 'film_tip', etc."""
    return station.name.lower()


# =================================================================
# PUNTOS
# =================================================================
@dataclass
class Punto:
    nombre: str
    z: float
    r: float


def construir_puntos(estaciones, p: Params):
    """Crea los puntos lógicos de cada estación. Nombre: p_<skey>_<nivel>."""
    puntos = {}
    for st in estaciones:
        sk = _skey(st)
        for nivel, r in niveles_de(st, p):
            nombre = f"p_{sk}_{nivel}"
            puntos[nombre] = Punto(nombre, st.z, r)
    return puntos


# =================================================================
# LÍNEAS
# =================================================================
@dataclass
class Linea:
    nombre: str
    p0: str       # punto inicial (clave)
    p1: str       # punto final (clave)
    rol: str      # AXIS/WALL/INLET/OUTLET/APERTURE/CHARGE/FAR/OUTER/FILM_*_INTERFACE/TOP
    tag: int      # tag físico, o None si no lleva
    izq: str      # región a la izquierda (diagnóstico)
    der: str      # región a la derecha  (diagnóstico)


# --- rol y nombre de cada banda de una cara vertical ---
_ROLES_FACE = ["APERTURE", "CHARGE", "FAR", "OUTER"]  # de adentro hacia afuera


def _tag_de_banda(rol, station: Station):
    if rol == "CHARGE":
        return TAGS["CHARGE_ZONE_TIP"] if station.name == "TIP" else TAGS["CHARGE_ZONE_BASE"]
    return None


def _linea_vertical(station, banda_idx, nivel0, nivel1, regL, regR, p: Params):
    """Construye la Linea de una banda vertical (ya se sabe que existe).
    Asigna nombre/rol/tag/dirección espejando el dibujador tip."""
    sk = _skey(station)
    n0_name, r0 = nivel0
    n1_name, r1 = nivel1

    if station.name == "INLET":
        # axis → top (sube)
        return Linea("l_inlet", f"p_{sk}_axis", f"p_{sk}_top",
                     "INLET", TAGS["INLET"], regL, regR)
    if station.name == "OUTLET":
        # top → axis (baja)
        return Linea("l_outlet", f"p_{sk}_top", f"p_{sk}_axis",
                     "OUTLET", TAGS["OUTLET"], regL, regR)
    if station.name == "FILM_TIP":
        return Linea("l_film_tip_face", f"p_{sk}_top", f"p_{sk}_axis",
                     "FILM_TIP_INTERFACE", TAGS["FILM_TIP_INTERFACE"], regL, regR)
    if station.name == "FILM_BASE":
        return Linea("l_film_base_face", f"p_{sk}_top", f"p_{sk}_axis",
                     "FILM_BASE_INTERFACE", TAGS["FILM_BASE_INTERFACE"], regL, regR)

    # Caras de membrana TIP / BASE: 4 bandas
    rol = _ROLES_FACE[banda_idx]
    tag = _tag_de_banda(rol, station)
    nombre = f"l_{sk}_{rol.lower()}"
    if station.name == "TIP":
        # TIP se recorre HACIA ABAJO (top → axis): de nivel1 (mayor r) a nivel0
        return Linea(nombre, f"p_{sk}_{n1_name}", f"p_{sk}_{n0_name}",
                     rol, tag, regL, regR)
    else:  # BASE: HACIA ARRIBA (axis → top): de nivel0 (menor r) a nivel1
        return Linea(nombre, f"p_{sk}_{n0_name}", f"p_{sk}_{n1_name}",
                     rol, tag, regL, regR)


def construir_lineas(estaciones, slabs, p: Params):
    """Deriva TODAS las líneas. Aplica la regla región_izq ≠ región_der para
    las verticales; agrega axis (1 por slab), top (1 por slab no-canal) y wall."""
    lineas = []

    # --- 1) LÍNEAS VERTICALES (caras): regla de existencia por banda ---
    for i, st in enumerate(estaciones):
        izq = slabs[i - 1] if i > 0 else None
        der = slabs[i]     if i < len(slabs) else None
        niveles = niveles_de(st, p)
        for j in range(len(niveles) - 1):
            n0, n1 = niveles[j], niveles[j + 1]
            rmid = 0.5 * (n0[1] + n1[1])
            regL = region_de(izq, rmid, st.z, p)
            regR = region_de(der, rmid, st.z, p)
            if regL != regR:
                lineas.append(_linea_vertical(st, j, n0, n1, regL, regR, p))

    # --- 2) EJE (r=0): una línea por slab, recorrida en z DECRECIENTE ---
    for s in slabs:
        suf = slab_suffix(s)
        # der.axis → izq.axis
        lineas.append(Linea(f"l_axis_{suf}",
                            f"p_{_skey(s.der)}_axis", f"p_{_skey(s.izq)}_axis",
                            "AXIS", TAGS["AXIS"], None, None))

    # --- 3) TOP (r=R_res): una por slab NO-canal, recorrida en z CRECIENTE ---
    for s in slabs:
        if s.name == "CANAL":
            continue   # sobre el canal hay membrana, no techo de fluido
        suf = slab_suffix(s)
        lineas.append(Linea(f"l_top_{suf}",
                            f"p_{_skey(s.izq)}_top", f"p_{_skey(s.der)}_top",
                            "TOP", None, None, None))

    # --- 4) WALL (pared del cono): tip_mouth → base_mouth ---
    lineas.append(Linea("l_wall", "p_tip_mouth", "p_base_mouth",
                        "WALL", TAGS["WALL"], FLUID, VOID))

    return lineas


# =================================================================
# SUPERFICIES (para el dump/preview; fusión de FLUID + cada FILM)
# =================================================================
def construir_superficies(slabs):
    sups, corrida = [], []
    for s in slabs:
        if s.region == FLUID:
            corrida.append(s)
        else:
            if corrida:
                sups.append(("DOMAIN_FLUID", list(corrida))); corrida = []
            tag = "DOMAIN_FILM_TIP" if s.name == "FILM_TIP" else "DOMAIN_FILM_BASE"
            sups.append((tag, [s]))
    if corrida:
        sups.append(("DOMAIN_FLUID", list(corrida)))
    return sups


# =================================================================
# DUMP DE TEXTO (validación a ojo)
# =================================================================
def describir(p: Params, titulo=""):
    est   = construir_estaciones(p)
    slabs = construir_slabs(est, p)
    pts   = construir_puntos(est, p)
    lns   = construir_lineas(est, slabs, p)
    sups  = construir_superficies(slabs)

    print("\n" + "=" * 74)
    print(f"  CONFIG: {titulo}   (film_tip={p.include_film_tip}, film_base={p.include_film_base})")
    print("=" * 74)

    print("\n  ESTACIONES (z nm):  " +
          "  ".join(f"{s.name}={s.z*NM:.1f}" for s in est))

    print(f"\n  PUNTOS: {len(pts)}")

    print(f"\n  LÍNEAS: {len(lns)}")
    # Agrupar por categoría para leer cómodo
    verticales = [l for l in lns if l.rol in
                  ("INLET", "OUTLET", "FILM_TIP_INTERFACE", "FILM_BASE_INTERFACE",
                   "APERTURE", "CHARGE", "FAR", "OUTER")]
    ejes = [l for l in lns if l.rol == "AXIS"]
    tops = [l for l in lns if l.rol == "TOP"]
    wall = [l for l in lns if l.rol == "WALL"]

    print("    -- verticales (caras) --")
    for l in verticales:
        tagtxt = f"tag {l.tag}" if l.tag else "(sin tag)"
        print(f"      {l.nombre:<20} {l.p0:<18}→ {l.p1:<18} {l.rol:<10} {tagtxt}  [{l.izq}|{l.der}]")
    print("    -- eje (axis) --")
    for l in ejes:
        print(f"      {l.nombre:<20} {l.p0:<18}→ {l.p1:<18} tag {l.tag}")
    print("    -- techos (top) --")
    for l in tops:
        print(f"      {l.nombre:<20} {l.p0:<18}→ {l.p1:<18} (sin tag)")
    print("    -- pared --")
    for l in wall:
        print(f"      {l.nombre:<20} {l.p0:<18}→ {l.p1:<18} tag {l.tag}")

    print("\n  SUPERFICIES (fluido fusionado + cada film):")
    for tag_name, run in sups:
        tramos = " + ".join(f"{s.izq.name}→{s.der.name}" for s in run)
        print(f"    {tag_name:<16} (tag {TAGS[tag_name]:>2})  =  {tramos}")

    tags_presentes = sorted({l.tag for l in lns if l.tag} |
                            {TAGS[t] for t, _ in sups})
    print(f"\n  TAGS PRESENTES: {tags_presentes}")


# =================================================================
# PREVIEW MATPLOTLIB (reflejado en r=0)
# =================================================================
C_FLUIDO   = "#cfe8ff"
C_FILM     = "#ffd8a8"
C_MEMBRANA = "#d0d0d0"
C_ROL = {
    "AXIS":                "#888888",
    "WALL":                "#000000",
    "INLET":               "#1f77b4",
    "OUTLET":              "#d62728",
    "CHARGE":              "#9467bd",
    "FILM_TIP_INTERFACE":  "#ff7f0e",
    "FILM_BASE_INTERFACE": "#2ca02c",
    "TOP":                 "#bbbbbb",
    "APERTURE":            "#cccccc",
    "FAR":                 "#cccccc",
    "OUTER":               "#cccccc",
}


def _color_linea(l: Linea):
    return C_ROL.get(l.rol, "#999999")


def _lw_linea(l: Linea):
    if l.rol == "CHARGE":               return 4.0
    if l.rol in ("AXIS", "TOP"):        return 1.0
    if l.rol in ("APERTURE", "FAR", "OUTER"): return 1.2
    return 2.5


def preview(p: Params, titulo=""):
    est   = construir_estaciones(p)
    slabs = construir_slabs(est, p)
    pts   = construir_puntos(est, p)
    lns   = construir_lineas(est, slabs, p)

    def fill_rect(ax, z0, z1, color):
        ax.fill([z0*NM, z1*NM, z1*NM, z0*NM],
                [-p.R_res*NM, -p.R_res*NM, p.R_res*NM, p.R_res*NM],
                color=color, ec="none", zorder=1)

    def fill_cono(ax, z0, z1, color_fluido, color_memb):
        zs = np.linspace(z0, z1, p.N_PTS_WALL)
        rs = np.array([R_canal(p, z) for z in zs])
        # fluido (espejado)
        ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[rs, -rs[::-1]]*NM,
                color=color_fluido, ec="none", zorder=1)
        # membrana arriba y abajo
        ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[rs, np.full_like(rs, p.R_res)[::-1]]*NM,
                color=color_memb, ec="none", zorder=1)
        ax.fill(np.r_[zs, zs[::-1]]*NM, np.r_[-rs, np.full_like(rs, -p.R_res)[::-1]]*NM,
                color=color_memb, ec="none", zorder=1)

    def dibujar(ax):
        for s in slabs:
            if s.name == "CANAL":
                fill_cono(ax, s.z0, s.z1, C_FLUIDO, C_MEMBRANA)
            elif s.region == FILM:
                fill_rect(ax, s.z0, s.z1, C_FILM)
            else:
                fill_rect(ax, s.z0, s.z1, C_FLUIDO)
        for l in lns:
            z0, r0 = pts[l.p0].z, pts[l.p0].r
            z1, r1 = pts[l.p1].z, pts[l.p1].r
            if l.rol == "WALL":
                zs = np.linspace(z0, z1, p.N_PTS_WALL)
                rs = np.array([R_canal(p, z) for z in zs])
                zz, rr = zs, rs
            else:
                zz, rr = np.array([z0, z1]), np.array([r0, r1])
            col, lw = _color_linea(l), _lw_linea(l)
            ax.plot(zz*NM,  rr*NM, color=col, lw=lw, zorder=3, solid_capstyle="round")
            if not np.allclose(rr, 0.0):
                ax.plot(zz*NM, -rr*NM, color=col, lw=lw, zorder=3, solid_capstyle="round")
        ax.set_xlabel("z [nm]"); ax.set_ylabel("r [nm]")
        ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.3)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 10))
    dibujar(axA)
    axA.set_title(f"{titulo}  —  vista completa (reflejado en r=0)")
    axA.set_xlim(-0.05*p.z_outlet*NM, 1.05*p.z_outlet*NM)
    axA.set_ylim(-p.R_res*1.25*NM, p.R_res*1.25*NM)

    dibujar(axB)
    zA = p.z_film_tip if p.include_film_tip else p.z_tip
    zB = p.z_film_base if p.include_film_base else p.z_base
    margin = 0.3 * p.L_pore
    r_zoom = p.R_base + p.L_charge + p.L_far
    axB.set_title("zoom al canal / film(s)")
    axB.set_xlim((zA - margin)*NM, (zB + margin)*NM)
    axB.set_ylim(-r_zoom*1.3*NM, r_zoom*1.3*NM)

    fig.tight_layout()
    return fig


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    casos = [
        ("sin film",  Params(include_film_tip=False, include_film_base=False)),
        ("film tip",  Params(include_film_tip=True,  include_film_base=False)),
        ("film base", Params(include_film_tip=False, include_film_base=True)),
        ("dos films", Params(include_film_tip=True,  include_film_base=True)),
    ]

    # 1) Dump de texto de los 4 casos
    for titulo, p in casos:
        describir(p, titulo)

    # 2) Preview de los 4 casos (una ventana por caso; cerralas para terminar)
    for titulo, p in casos:
        preview(p, titulo)
    plt.show()