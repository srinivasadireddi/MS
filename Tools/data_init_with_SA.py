"""
Dataset initialization with ICON-like liquid saturation adjustment.

This module is intentionally separate from Tools/dataset_init.py. It delegates
the file opening to the existing initializer, then lazily replaces clw, hus,
and ta by fields adjusted with the Newton saturation-adjustment equations.
"""

import copy

import numpy as np
import xarray as xr

from Tools import dataset_init
from Tools.thermodynamics import C4LES, C5LES, TMELT, qsat_rho_water


# ICON-like thermodynamic constants. Values are standard dry/vapor/liquid/ice
# heat capacities at constant volume where applicable.
CVD = 717.6
CVV = 1407.95
CPV = 1869.46
CLW_HEAT = 4186.84
CI_HEAT = 2108.0
ALV = 2.5008e6
LVC = ALV - (CPV - CLW_HEAT) * TMELT    # LVC = 3'133'792.347

"""Ancient values
CVD = 717.6
CVV = 1407.95
CLW_HEAT = 4186.84
CI_HEAT = 2108.0
LVC = 2.5008e6"""

SA_REQUIRED_VARS = ("clw", "hus", "ta", "rho", "qr", "cli", "qg", "qs")


def dqsatdT_rho(qsat, tk):
    """Temperature derivative of qsat_rho_water at fixed density."""
    return qsat * (C5LES / (tk - C4LES) ** 2 - 1.0 / tk)


def saturation_adjustment_newton(qv, ql, tk, rho, qr=None, qti=None, n_iter=6):
    """
    ICON-like liquid saturation adjustment using Newton iterations.

    Returns adjusted `(qv, ql, tk)` while conserving the liquid-vapor internal
    energy part used by ICON's saturation_adjustment routine.
    """
    if qr is None:
        qr = xr.zeros_like(qv)
    if qti is None:
        qti = xr.zeros_like(qv)

    qt = qv + ql + qr + qti
    cvc = CVD * (1.0 - qt) + CLW_HEAT * qr + CI_HEAT * qti
    cv_initial = cvc + CVV * qv + CLW_HEAT * ql
    ue = cv_initial * tk - ql * LVC

    total_liquid_vapor = qv + ql
    tx_direct = ue / (cv_initial + ql * (CVV - CLW_HEAT))
    qsat_direct = qsat_rho_water(tx_direct, rho)
    evaporate_all = total_liquid_vapor <= qsat_direct

    tx = tk
    for _ in range(n_iter):
        qsat = qsat_rho_water(tx, rho)
        dqsat = dqsatdT_rho(qsat, tx)
        qcx = total_liquid_vapor - qsat
        cv = cvc + CVV * qsat + CLW_HEAT * qcx
        ux = cv * tx - qcx * LVC
        dux = cv + dqsat * (LVC + (CVV - CLW_HEAT) * tx)
        tx = tx - (ux - ue) / dux

    qsat_final = qsat_rho_water(tx, rho)
    qv_newton = qsat_final
    ql_newton = xr.where(total_liquid_vapor - qsat_final > 0.0, total_liquid_vapor - qsat_final, 0.0)

    qv_adjusted = xr.where(evaporate_all, total_liquid_vapor, qv_newton)
    ql_adjusted = xr.where(evaporate_all, 0.0, ql_newton)
    tk_adjusted = xr.where(evaporate_all, tx_direct, tx)

    qv_adjusted = qv_adjusted.rename(qv.name or "hus")
    ql_adjusted = ql_adjusted.rename(ql.name or "clw")
    tk_adjusted = tk_adjusted.rename(tk.name or "ta")
    return qv_adjusted, ql_adjusted, tk_adjusted


def ds_init(myJob, chunks="auto", only_2d=False, n_iter=6):
    """
    Initialize a dataset like dataset_init.ds_init, then apply Newton SA.

    `myJob.var_names` is respected for the returned dataset, but required
    auxiliary fields are loaded internally so the adjustment can use rain and
    total ice as in ICON.
    """
    original_var_names = list(myJob.var_names or [])
    job_for_load = copy.copy(myJob)
    job_for_load.var_names = sorted(set(original_var_names).union(SA_REQUIRED_VARS))

    ds = dataset_init.ds_init(job_for_load, chunks=chunks, only_2d=only_2d)

    qti = ds["cli"] + ds["qg"] + ds["qs"]
    qv_adjusted, ql_adjusted, tk_adjusted = saturation_adjustment_newton(
        qv=ds["hus"],
        ql=ds["clw"],
        tk=ds["ta"],
        rho=ds["rho"],
        qr=ds["qr"],
        qti=qti,
        n_iter=n_iter,
    )

    ds = ds.assign(
        hus=qv_adjusted,
        clw=ql_adjusted,
        ta=tk_adjusted,
    )
    ds["hus"].attrs.update({"saturation_adjusted": "newton"})
    ds["clw"].attrs.update({"saturation_adjusted": "newton"})
    ds["ta"].attrs.update({"saturation_adjusted": "newton"})
    ds.attrs["saturation_adjustment"] = "newton_liquid_water_energy_conserving"
    ds.attrs["saturation_adjustment_iterations"] = n_iter

    if original_var_names:
        keep = [name for name in original_var_names if name in ds]
        for required_name in ("hus", "clw", "ta"):
            if required_name in ds and required_name not in keep:
                keep.append(required_name)
        ds = ds[keep]

    return ds
