"""
Basic thermodynamic helper functions reused across diagnostics scripts.

The formulas below match the water-saturation expressions already used in the
histogram scripts and are intended to stay close to the ICON implementation
used in this project.
"""

import numpy as np


# Physical constants (from mo_physical_constants and mo_aes_thermo)
RV = 461.51
TMELT = 273.15
C1ES = 610.78
C3LES = 17.269
C4LES = 35.86
C5LES = C3LES * (TMELT - C4LES)


def sat_pres_water(TK):
    """
    Saturation vapor pressure over liquid water [Pa].

    Parameters
    ----------
    TK : array_like
        Temperature in Kelvin.
    """
    return C1ES * np.exp(C3LES * (TK - TMELT) / (TK - C4LES))


def qsat_rho_water(TK, rho):
    """
    Saturation specific humidity at constant density [kg/kg].

    Parameters
    ----------
    TK : array_like
        Temperature in Kelvin.
    rho : array_like
        Air density in kg/m^3.
    """
    return sat_pres_water(TK) / (rho * RV * TK)


def relative_humidity_water(qv, TK, rho):
    """
    Relative humidity over liquid water as a fraction.

    Parameters
    ----------
    qv : array_like
        Water vapor specific humidity [kg/kg].
    TK : array_like
        Temperature in Kelvin.
    rho : array_like
        Air density in kg/m^3.
    """
    return qv / qsat_rho_water(TK, rho)


def adjust_liquid_vapor_to_saturation(qv, ql, TK, rho):
    """
    Saturation adjustment at fixed temperature using the liquid-vapor pool only.

    The adjustment conserves qv + ql and does not modify temperature.
    If the state is supersaturated, excess vapor condenses into liquid.
    If the state is subsaturated and liquid is available, liquid evaporates
    until saturation is reached or all liquid is depleted.

    Parameters
    ----------
    qv : array_like
        Water vapor specific humidity [kg/kg].
    ql : array_like
        Cloud liquid water specific humidity [kg/kg].
    TK : array_like
        Temperature in Kelvin.
    rho : array_like
        Air density in kg/m^3.

    Returns
    -------
    tuple
        (qv_adjusted, ql_adjusted)
    """
    qsat = qsat_rho_water(TK, rho)
    total_liquid_vapor = qv + ql

    # Keep vapor at saturation whenever the combined pool allows it.
    qv_adjusted = np.minimum(total_liquid_vapor, qsat)
    ql_adjusted = np.maximum(total_liquid_vapor - qsat, 0.0)
    return qv_adjusted, ql_adjusted
