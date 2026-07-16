# -*- coding: utf-8 -*-
"""
Conversor: estado del dibujador (dict liviano) → Params (dataclass de Capa 1).
Único punto de traducción entre la GUI y las capas. Mantenerlo acá evita que
el dibujador "sepa" de Params y que las capas "sepan" del dict de la GUI.
"""

from capa1_modelo import Params

# Defaults de corona cuando el usuario NO activó la zona de carga. La Capa 1
# necesita L_charge/L_far para construir la cara de membrana (los breakpoints
# charge/far/outer existen siempre en la topología actual). Con corona
# desactivada usamos una zona de carga mínima pero no nula.
L_CHARGE_DEFAULT = 2e-9
L_FAR_DEFAULT    = 2e-9


def estado_a_params(st) -> Params:
    """Convierte el dict de estado del dibujador en un Params construible.

    Requiere que el estado tenga reservorios definidos (L_res/R_res no None);
    si no, lanza ValueError (la geometría no es generable sin reservorios)."""
    if not st.get("L_res") or not st.get("R_res"):
        raise ValueError("Missing reservoirs (L_res/R_res): the geometry "
                         "is not generable yet.")

    usar_corona = bool(st.get("usar_corona")) and st.get("L_charge", 0) > 0
    L_charge = st["L_charge"] if usar_corona else L_CHARGE_DEFAULT
    L_far    = st.get("L_far", 0.0) if usar_corona else L_FAR_DEFAULT

    ft, fb = st.get("film_tip"), st.get("film_base")

    # Tipo de perfil de pared. El estado guarda "cylinder"/"conical"/"bullet";
    # Params solo distingue "bullet" del resto (lineal), pero pasamos el valor
    # tal cual para que el JSON registre el tipo real. h_param solo importa si
    # es bullet; si no viene, Params usa su default.
    canal_tipo = st.get("type", "conical")
    h_param = st.get("h_param")

    kwargs = dict(
        L_pore = st["L_pore"],
        D_tip  = st["D_tip"],
        D_base = st["D_base"],     # cilindro: ya viene == D_tip desde el estado
        L_res  = st["L_res"],
        R_res  = st["R_res"],
        L_charge = L_charge,
        L_far    = L_far,
        include_film_tip  = ft is not None,
        delta_film_tip    = (ft["delta"] if ft else 10e-9),
        include_film_base = fb is not None,
        delta_film_base   = (fb["delta"] if fb else 10e-9),
        channel_type = canal_tipo,
        # N_PTS_WALL queda en su default (no lo expone el dibujador).
    )
    if canal_tipo == "bullet" and h_param:
        kwargs["h_param"] = h_param
    return Params(**kwargs)