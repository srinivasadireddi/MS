#!/usr/bin/env python3
"""
Evaluate the phase-partition scheme on ORCESTRA curtain files.

This script replaces the notebook/XGBoost workflow with the partition model
defined in Tools/partition_functions.py and evaluates curtains against the
global-mean reference used by compiling_scripts/compute_partition_R2.py.

The main output is a figure with curtain dates on the x axis and mass-weighted
R^2 on the y axis. A CSV summary is also written to disk.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-curtains-romain")

import sys
import re
import argparse
import fnmatch
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager


current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_folder, ".."))

from Tools.thermodynamics import relative_humidity_water
from Tools.data_init_with_SA import saturation_adjustment_newton
from Tools.partition_functions import (
    QG_QGS_LAZY_MODE,
    compute_Pi_l_C_lazy,
    compute_Pi_r_P_lazy,
    compute_Pi_g_P_lazy,
    compute_Pi_g_gs_lazy,
)


DEFAULT_CURTAIN_DIR = "/work/mh0492/m301067/ec-curtains/data/curtains/build-master-031224-new-rain"
DEFAULT_GLOB = "orcestra_ec-curtain_*.nc"
DEFAULT_QMIN = 1e-12
DEFAULT_HEIGHT_MIN = 40.0
DEFAULT_HEIGHT_MAX = 90.0
DEFAULT_CELL_AREA_KM2 = 1.25
DEFAULT_PARTITION_SUMMARY = os.path.join(path_to_output_data := os.path.join(current_folder, "../Data/model_evaluation"), "partition_R2_summary.txt")

path_to_figures = os.path.join(current_folder, "../figures/curtains_romain")
path_to_stix_fonts = os.path.join(current_folder, "../../stixfonts/source")

SPECIES_CONFIGS = OrderedDict(
    [
        ("l", {"hydro_var": "qc", "group_var": "qC", "label": "cloud liquid", "curve_label": r"$q_c/q_{\mathscr{C}}$", "role": "global"}),
        ("i", {"hydro_var": "qi", "group_var": "qC", "label": "cloud ice", "curve_label": r"$q_i/q_{\mathscr{C}}$", "role": "global"}),
        ("r", {"hydro_var": "qr", "group_var": "qP", "label": "rain", "curve_label": r"$q_r/q_{\mathscr{P}}$", "role": "global"}),
        ("g_P", {"hydro_var": "qg", "group_var": "qP", "label": "graupel predictive qP", "curve_label": r"$q_g/q_{\mathscr{P}}$", "role": "global"}),
        ("s_P", {"hydro_var": "qs", "group_var": "qP", "label": "snow predictive qP", "curve_label": r"$q_s/q_{\mathscr{P}}$", "role": "global"}),
        ("g_gs", {"hydro_var": "qg", "group_var": "qgs", "label": "graupel diagnostic qgs", "curve_label": r"$q_g/(q_g+q_s)$", "role": "diagnostic"}),
        ("s_gs", {"hydro_var": "qs", "group_var": "qgs", "label": "snow diagnostic qgs", "curve_label": r"$q_s/(q_g+q_s)$", "role": "diagnostic"}),
    ]
)
GLOBAL_SPECIES = tuple(name for name, cfg in SPECIES_CONFIGS.items() if cfg["role"] == "global")

SPECIES_LINE_STYLES = {
    "l": {"color": "#5209cf", "marker": "o", "linewidth": 1.9, "markersize": 5},
    "i": {"color": "#ff0000", "marker": "D", "linewidth": 1.8, "markersize": 4.8},
    "r": {"color": "#16a600", "marker": "s", "linewidth": 1.9, "markersize": 5},
    "g_P": {"color": "#eb8021", "marker": "^", "linewidth": 1.9, "markersize": 5},
    "s_P": {"color": "#00a3bf", "marker": "v", "linewidth": 1.8, "markersize": 4.8},
    "g_gs": {"color": "#a85200", "marker": "P", "linewidth": 1.8, "markersize": 4.8},
    "s_gs": {"color": "#00738a", "marker": "X", "linewidth": 1.8, "markersize": 4.8},
}


def progress(message):
    print(message, flush=True)


def configure_fonts():
    """Prefer local STIX Two fonts when available."""
    font_paths = [
        os.path.join(path_to_stix_fonts, "STIXTwoText-Regular.input.ttf"),
        os.path.join(path_to_stix_fonts, "STIXTwoText-Italic.input.ttf"),
        os.path.join(path_to_stix_fonts, "STIXTwoText-Bold.input.ttf"),
        os.path.join(path_to_stix_fonts, "STIXTwoText-BoldItalic.input.ttf"),
        os.path.join(path_to_stix_fonts, "STIXTwoMath.input.ttf"),
    ]
    existing_paths = [path for path in font_paths if os.path.exists(path)]

    for path in existing_paths:
        font_manager.fontManager.addfont(path)

    if len(existing_paths) == len(font_paths):
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "STIX Two Text",
                "font.serif": ["STIX Two Text"],
                "font.size": 13,
                "axes.titlesize": 13,
                "axes.labelsize": 13,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 11,
                "figure.titlesize": 13,
                "mathtext.fontset": "custom",
                "mathtext.rm": "STIX Two Text",
                "mathtext.it": "STIX Two Text:italic",
                "mathtext.bf": "STIX Two Text:bold",
                "mathtext.cal": "STIX Two Math",
            }
        )
        return

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 13,
            "mathtext.fontset": "custom",
            "mathtext.rm": "DejaVu Serif",
            "mathtext.it": "DejaVu Serif:italic",
            "mathtext.bf": "DejaVu Serif:bold",
            "mathtext.cal": "STIXGeneral:italic",
        }
    )


def parse_species(value):
    value = value.strip()
    if value.lower() == "all":
        return list(SPECIES_CONFIGS)

    aliases = {
        "g_p": "g_P",
        "s_p": "s_P",
        "g_gs": "g_gs",
        "s_gs": "s_gs",
    }
    species = [aliases.get(item.strip().lower(), item.strip()) for item in value.split(",") if item.strip()]
    unknown = sorted(set(species) - set(SPECIES_CONFIGS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown species: {', '.join(unknown)}")
    return species


def parse_args():
    parser = argparse.ArgumentParser(description="Compute mass-weighted curtain R2 with the partition model.")
    parser.add_argument("--curtain_dir", default=DEFAULT_CURTAIN_DIR, help="Directory containing curtain NetCDF files.")
    parser.add_argument("--glob_pattern", default=DEFAULT_GLOB, help="Filename pattern used to select curtains.")
    parser.add_argument("--species", type=parse_species, default=parse_species("all"), help="Comma-separated species list or 'all'.")
    parser.add_argument("--qmin", type=float, default=DEFAULT_QMIN, help="Minimum pool mass fraction used to keep a cell.")
    parser.add_argument("--height_min", type=float, default=DEFAULT_HEIGHT_MIN, help="Lowest full level to keep.")
    parser.add_argument("--height_max", type=float, default=DEFAULT_HEIGHT_MAX, help="Highest full level to keep.")
    parser.add_argument(
        "--time_mode",
        choices=("nearest_track", "nearest_file", "first"),
        default="nearest_track",
        help="How to pick the model time inside each curtain file.",
    )
    parser.add_argument(
        "--cell_area_km2",
        type=float,
        default=DEFAULT_CELL_AREA_KM2,
        help="Horizontal cell area used in the mass weight. It scales all weights equally, so R2 is unchanged if this constant is adjusted.",
    )
    parser.add_argument("--summary_csv", default=os.path.join(path_to_output_data, "curtains_romain_summary.csv"))
    parser.add_argument("--output_png", default=os.path.join(path_to_figures, "fig_R2_vs_date.png"))
    parser.add_argument("--output_pdf", default=os.path.join(path_to_figures, "fig_R2_vs_date.pdf"))
    parser.add_argument(
        "--partition_summary",
        default=DEFAULT_PARTITION_SUMMARY,
        help="Text summary produced by compute_partition_R2.py. The global mean_fraction for each species is read from this file and used as the curtain R2 reference.",
    )
    parser.add_argument(
        "--no_saturation_adjustment",
        action="store_true",
        help="Use raw qv/qc/ta from curtain files instead of applying the ICON-like Newton saturation adjustment.",
    )
    return parser.parse_args()


def parse_curtain_datetime(file_name):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})", file_name)
    if match is None:
        return pd.NaT
    date_part, hour_part, minute_part = match.groups()
    return pd.Timestamp(f"{date_part} {hour_part}:{minute_part}:00")


def get_curtain_files(curtain_dir, glob_pattern):
    curtain_dir = os.path.abspath(curtain_dir)
    if not os.path.isdir(curtain_dir):
        raise FileNotFoundError(f"Curtain directory not found: {curtain_dir}")

    file_names = sorted(
        name for name in os.listdir(curtain_dir)
        if fnmatch.fnmatch(name, glob_pattern)
    )
    if not file_names:
        raise FileNotFoundError(f"No curtain file matches '{glob_pattern}' in {curtain_dir}")
    return [os.path.join(curtain_dir, name) for name in file_names]


def choose_time_index(ds, time_mode):
    model_time = ds["time"].values.astype("datetime64[ns]").astype("int64")
    track_time = ds["track_time"].values.astype("datetime64[ns]").astype("int64")

    if time_mode == "first":
        indices = np.zeros(ds.sizes["track"], dtype=int)
    elif time_mode == "nearest_file":
        center = np.int64(np.median(track_time))
        nearest_index = int(np.abs(model_time - center).argmin())
        indices = np.full(ds.sizes["track"], nearest_index, dtype=int)
    else:
        indices = np.abs(model_time[:, None] - track_time[None, :]).argmin(axis=0).astype(int)

    unique_indices = np.unique(indices)
    selected_times = ds["time"].values[unique_indices].astype("datetime64[ns]")
    return indices, selected_times


def select_var_at_time(ds, var_name, time_indices):
    da = ds[var_name]
    if "time" not in da.dims:
        return da
    indexer = xr.DataArray(time_indices, dims=("track",))
    return da.isel(time=indexer)


def transpose_if_needed(da):
    if da.dims == ("track", "height_full"):
        return da.transpose("height_full", "track")
    if da.dims == ("track", "height_half"):
        return da.transpose("height_half", "track")
    return da


def compute_dz_from_zghalf(zghalf, height_full):
    zghalf_values = np.asarray(zghalf.values, dtype=float)
    dz_values = zghalf_values[:-1, :] - zghalf_values[1:, :]
    return xr.DataArray(
        dz_values,
        dims=("height_full", "track"),
        coords={"height_full": height_full.values},
        name="dzghalf",
    )


def apply_saturation_adjustment_to_curtain(curtain):
    qti = curtain["qi"] + curtain["qg"] + curtain["qs"]
    qv_adjusted, qc_adjusted, ta_adjusted = saturation_adjustment_newton(
        qv=curtain["qv"],
        ql=curtain["qc"],
        tk=curtain["ta"],
        rho=curtain["rho"],
        qr=curtain["qr"],
        qti=qti,
        n_iter=6,
    )

    curtain = curtain.assign(
        qv=qv_adjusted,
        qc=qc_adjusted,
        ta=ta_adjusted,
    )
    curtain["qv"].attrs.update({"saturation_adjusted": "newton"})
    curtain["qc"].attrs.update({"saturation_adjusted": "newton"})
    curtain["ta"].attrs.update({"saturation_adjusted": "newton"})
    curtain.attrs["saturation_adjustment"] = "newton_liquid_water_energy_conserving"
    curtain.attrs["saturation_adjustment_iterations"] = 6
    return curtain


def prepare_curtain_slice(ds, time_mode, height_min, height_max, apply_saturation_adjustment=True):
    time_indices, selected_times = choose_time_index(ds, time_mode)
    required_vars = ["qv", "qc", "qi", "qs", "qg", "qr", "ta", "rho", "zghalf", "track_time"]
    if QG_QGS_LAZY_MODE == 5:
        required_vars.extend(["cllvi", "qivi"])

    selected = xr.Dataset(attrs=ds.attrs)
    for var_name in required_vars:
        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' is missing from {ds.encoding.get('source', 'curtain file')}")
        selected[var_name] = transpose_if_needed(select_var_at_time(ds, var_name, time_indices))

    selected = selected.sel(
        height_full=slice(height_min, height_max),
        height_half=slice(height_min, height_max + 1.0),
    )

    selected["dzghalf"] = compute_dz_from_zghalf(selected["zghalf"], selected["height_full"])
    if apply_saturation_adjustment:
        selected = apply_saturation_adjustment_to_curtain(selected)

    meta = {
        "time_mode": time_mode,
        "saturation_adjustment": selected.attrs.get("saturation_adjustment", "none"),
        "selected_time_start": pd.Timestamp(selected_times.min()),
        "selected_time_end": pd.Timestamp(selected_times.max()),
        "n_unique_model_times": int(selected_times.size),
        "track_time_start": pd.Timestamp(selected["track_time"].values[0]),
        "track_time_end": pd.Timestamp(selected["track_time"].values[-1]),
    }
    return selected, meta


def compute_chunk_moments(omega_y, frac_true, err2):
    dataset = xr.Dataset(
        {
            "Sum_w": omega_y,
            "Sum_wy": omega_y * frac_true,
            "Sum_wy2": omega_y * frac_true ** 2,
            "Sum_werr2": omega_y * err2,
        }
    ).sum(skipna=True)
    return {name: float(dataset[name].item()) for name in dataset.data_vars}


def load_reference_mean_fractions(summary_path, active_species):
    summary_path = os.path.abspath(summary_path)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Partition summary not found: {summary_path}")

    current_species = None
    mean_fractions = {}
    with open(summary_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            species_match = re.match(r"^([A-Za-z0-9_]+):\s", line)
            if species_match:
                current_species = species_match.group(1)
                continue

            mean_match = re.match(r"^mean_fraction\s*=\s*([0-9eE+\-.]+)$", line)
            if mean_match and current_species is not None:
                mean_fractions[current_species] = float(mean_match.group(1))

    missing = [name for name in active_species if name not in mean_fractions]
    if missing:
        raise ValueError(
            f"Missing mean_fraction entries for species {', '.join(missing)} in {summary_path}"
        )

    # Keep the reference climatology global, but make complementary pairs exact.
    # This preserves the R2 symmetry for pairs evaluated with the same pool,
    # mask, weight, and complementary predictions.
    if "l" in mean_fractions:
        mean_fractions["i"] = 1.0 - mean_fractions["l"]
    if "g_gs" in mean_fractions:
        mean_fractions["s_gs"] = 1.0 - mean_fractions["g_gs"]

    return mean_fractions


def compute_species_diagnostics(curtain, active_species, qmin, cell_area_m2, reference_mean_fractions):
    qC = curtain["qc"] + curtain["qi"]
    qP = curtain["qr"] + curtain["qg"] + curtain["qs"]
    qgs = curtain["qg"] + curtain["qs"]
    RH = relative_humidity_water(curtain["qv"], curtain["ta"], curtain["rho"])
    viqC = curtain["cllvi"] + curtain["qivi"] if QG_QGS_LAZY_MODE == 5 else None

    geom_mass = curtain["rho"] * curtain["dzghalf"] * cell_area_m2
    true_fields = {
        "l": curtain["qc"],
        "i": curtain["qi"],
        "r": curtain["qr"],
        "g_P": curtain["qg"],
        "s_P": curtain["qs"],
        "g_gs": curtain["qg"],
        "s_gs": curtain["qs"],
    }
    pool_fields = {"qC": qC, "qP": qP, "qgs": qgs}

    pred_fields = {}
    qr_pred = None
    if any(name in active_species for name in ("l", "i")):
        ql_pred = compute_Pi_l_C_lazy(
            qC,
            curtain["ta"],
            RH=RH,
            qv=curtain["qv"],
            ql=curtain["qc"],
            rho=curtain["rho"],
        )
        pred_fields["l"] = ql_pred
        pred_fields["i"] = qC - ql_pred
    if "r" in active_species or any(name in active_species for name in ("g_P", "s_P")):
        qr_pred = compute_Pi_r_P_lazy(qP, curtain["ta"])
        if "r" in active_species:
            pred_fields["r"] = qr_pred
    if any(name in active_species for name in ("g_P", "s_P")):
        qg_pred_P = compute_Pi_g_P_lazy(
            qP,
            curtain["ta"],
            RH=RH,
            qC=qC,
            viqC=viqC,
            qr_pred=qr_pred,
        )
        pred_fields["g_P"] = qg_pred_P
        qs_pred_P = (qP - qr_pred - qg_pred_P).where((qP - qr_pred - qg_pred_P) > 0.0, 0.0)
        pred_fields["s_P"] = qs_pred_P
    if any(name in active_species for name in ("g_gs", "s_gs")):
        qg_pred_gs = compute_Pi_g_gs_lazy(qgs, curtain["ta"], qC=qC, RH=RH, qP=qP, viqC=viqC)
        pred_fields["g_gs"] = qg_pred_gs
        pred_fields["s_gs"] = (qgs - qg_pred_gs).where((qgs - qg_pred_gs) > 0.0, 0.0)

    diagnostics = OrderedDict()
    for name in active_species:
        cfg = SPECIES_CONFIGS[name]
        qx_true = true_fields[name]
        qx_pred = pred_fields[name]
        qY_true = pool_fields[cfg["group_var"]]
        mask = qY_true > qmin

        frac_true = (qx_true / qY_true).where(mask)
        frac_pred = (qx_pred / qY_true).where(mask)
        omega_y = (geom_mass * qY_true).where(mask)
        err2 = (frac_true - frac_pred) ** 2
        moments = compute_chunk_moments(omega_y, frac_true, err2)

        sum_w = moments["Sum_w"]
        sum_wy = moments["Sum_wy"]
        sum_wy2 = moments["Sum_wy2"]
        sum_werr2 = moments["Sum_werr2"]
        valid_cells = int(mask.sum().item())
        reference_mean_fraction = reference_mean_fractions[name]

        if sum_w <= 0.0:
            curtain_mean_fraction = np.nan
            denominator = np.nan
            local_denominator = np.nan
            r2_value = np.nan
            local_r2_value = np.nan
        else:
            curtain_mean_fraction = sum_wy / sum_w
            local_denominator = (
                sum_wy2
                - 2.0 * curtain_mean_fraction * sum_wy
                + sum_w * curtain_mean_fraction ** 2
            )
            denominator = (
                sum_wy2
                - 2.0 * reference_mean_fraction * sum_wy
                + sum_w * reference_mean_fraction ** 2
            )
            r2_value = np.nan if denominator <= 0.0 else 1.0 - sum_werr2 / denominator
            local_r2_value = np.nan if local_denominator <= 0.0 else 1.0 - sum_werr2 / local_denominator

        diagnostics[name] = {
            "valid_cells": valid_cells,
            "omega_y_tot": sum_w,
            "omega_x_tot": sum_wy,
            "curtain_mean_fraction": curtain_mean_fraction,
            "reference_mean_fraction": reference_mean_fraction,
            "numerator": sum_werr2,
            "denominator": denominator,
            "local_denominator": local_denominator,
            "R2": r2_value,
            "local_R2": local_r2_value,
            "baseline_R2": 0.0 if np.isfinite(denominator) and denominator > 0.0 else np.nan,
        }

    global_num = 0.0
    global_den = 0.0
    for name, diag in diagnostics.items():
        if (
            SPECIES_CONFIGS[name]["role"] == "global"
            and np.isfinite(diag["numerator"])
            and np.isfinite(diag["denominator"])
            and diag["denominator"] > 0.0
        ):
            global_num += diag["numerator"]
            global_den += diag["denominator"]
    global_r2 = np.nan if global_den <= 0.0 else 1.0 - global_num / global_den
    return diagnostics, global_r2


def build_summary_row(file_path, curtain_datetime, meta, diagnostics, global_r2):
    row = {
        "file_name": os.path.basename(file_path),
        "curtain_datetime": curtain_datetime,
        "time_mode": meta["time_mode"],
        "track_time_start": meta["track_time_start"],
        "track_time_end": meta["track_time_end"],
        "selected_time_start": meta["selected_time_start"],
        "selected_time_end": meta["selected_time_end"],
        "n_unique_model_times": meta["n_unique_model_times"],
        "QG_QGS_LAZY_MODE": QG_QGS_LAZY_MODE,
        "global_R2": global_r2,
    }

    for name, diag in diagnostics.items():
        row[f"{name}_valid_cells"] = diag["valid_cells"]
        row[f"{name}_omega_y_tot"] = diag["omega_y_tot"]
        row[f"{name}_omega_x_tot"] = diag["omega_x_tot"]
        row[f"{name}_curtain_mean_fraction"] = diag["curtain_mean_fraction"]
        row[f"{name}_reference_mean_fraction"] = diag["reference_mean_fraction"]
        row[f"{name}_numerator"] = diag["numerator"]
        row[f"{name}_denominator"] = diag["denominator"]
        row[f"{name}_local_denominator"] = diag["local_denominator"]
        row[f"{name}_R2"] = diag["R2"]
        row[f"{name}_local_R2"] = diag["local_R2"]
        row[f"{name}_baseline_R2"] = diag["baseline_R2"]

    return row


def style_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def compute_across_curtain_summed_r2(summary_df, active_species):
    aggregated = OrderedDict()
    total_num = 0.0
    total_den = 0.0

    for name in active_species:
        numerators = summary_df[f"{name}_numerator"].to_numpy(dtype=float)
        denominators = summary_df[f"{name}_denominator"].to_numpy(dtype=float)
        valid = np.isfinite(numerators) & np.isfinite(denominators) & (denominators > 0.0)

        if not np.any(valid):
            aggregated[name] = np.nan
            continue

        numerator = float(np.sum(numerators[valid]))
        denominator = float(np.sum(denominators[valid]))
        aggregated[name] = np.nan if denominator <= 0.0 else 1.0 - numerator / denominator
        if SPECIES_CONFIGS[name]["role"] == "global":
            total_num += numerator
            total_den += denominator

    aggregated["total"] = np.nan if total_den <= 0.0 else 1.0 - total_num / total_den
    return aggregated


def plot_r2_vs_date(summary_df, active_species, output_png, output_pdf):
    configure_fonts()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    aggregated_r2 = compute_across_curtain_summed_r2(summary_df, active_species)
    fig, ax = plt.subplots(figsize=(14.0, 5.8), dpi=300)
    dates = pd.to_datetime(summary_df["curtain_datetime"])

    plotted_species = list(active_species)
    for name in plotted_species:
        aggregated_value = aggregated_r2[name]
        label = (
            f"{SPECIES_CONFIGS[name]['label']} (R²_tot={aggregated_value:.3f})"
            if np.isfinite(aggregated_value)
            else f"{SPECIES_CONFIGS[name]['label']} (R²_tot=nan)"
        )
        style = SPECIES_LINE_STYLES.get(name, {})
        linestyle = "--" if SPECIES_CONFIGS[name]["role"] == "diagnostic" else "-"
        ax.plot(
            dates,
            summary_df[f"{name}_R2"].to_numpy(dtype=float),
            color=style.get("color", "#666666"),
            marker=style.get("marker", "o"),
            linewidth=style.get("linewidth", 1.8),
            markersize=style.get("markersize", 4.8),
            linestyle=linestyle,
            label=label,
            zorder=3,
        )

    total_value = aggregated_r2["total"]
    total_label = (
        f"total summed (R²_tot={total_value:.3f})"
        if np.isfinite(total_value)
        else "total summed (R²_tot=nan)"
    )
    ax.plot(
        dates,
        summary_df["global_R2"].to_numpy(dtype=float),
        color="#111111",
        marker="o",
        linewidth=2.3,
        markersize=5.5,
        label=total_label,
        zorder=4,
    )

    style_axes(ax)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_ylabel(r"$R^2$")
    ax.set_xlabel("Curtain date")
    ax.set_title(r"Curtain $R^2$ by date")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    values_for_limits = [summary_df["global_R2"].to_numpy(dtype=float)]
    values_for_limits.extend(summary_df[f"{name}_R2"].to_numpy(dtype=float) for name in plotted_species)
    finite_values = np.concatenate([values[np.isfinite(values)] for values in values_for_limits if np.isfinite(values).any()])
    if finite_values.size:
        ymin = min(-0.05, float(np.nanmin(finite_values)) - 0.05)
        ymax = max(1.05, float(np.nanmax(finite_values)) + 0.05)
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(path_to_output_data, exist_ok=True)
    os.makedirs(path_to_figures, exist_ok=True)

    cell_area_m2 = args.cell_area_km2 * 1e6
    reference_mean_fractions = load_reference_mean_fractions(args.partition_summary, args.species)
    curtain_files = get_curtain_files(args.curtain_dir, args.glob_pattern)
    progress(f"Found {len(curtain_files)} curtain files in {args.curtain_dir}")
    progress(f"Active species: {', '.join(args.species)}")
    progress(f"time_mode={args.time_mode}, qmin={args.qmin:.3e}, cell_area_km2={args.cell_area_km2:.6f}")
    progress(f"Loaded global reference means from {args.partition_summary}")
    progress(
        "Reference mean fractions: "
        + ", ".join(f"{name}={reference_mean_fractions[name]:.6f}" for name in args.species)
    )

    summary_rows = []
    for index, file_path in enumerate(curtain_files, start=1):
        file_name = os.path.basename(file_path)
        progress(f"[{index}/{len(curtain_files)}] Processing {file_name}")

        with xr.open_dataset(file_path) as ds:
            curtain, meta = prepare_curtain_slice(
                ds=ds,
                time_mode=args.time_mode,
                height_min=args.height_min,
                height_max=args.height_max,
                apply_saturation_adjustment=not args.no_saturation_adjustment,
            )
            diagnostics, global_r2 = compute_species_diagnostics(
                curtain=curtain,
                active_species=args.species,
                qmin=args.qmin,
                cell_area_m2=cell_area_m2,
                reference_mean_fractions=reference_mean_fractions,
            )

        curtain_datetime = parse_curtain_datetime(file_name)
        if pd.isna(curtain_datetime):
            curtain_datetime = meta["track_time_start"]

        row = build_summary_row(
            file_path=file_path,
            curtain_datetime=curtain_datetime,
            meta=meta,
            diagnostics=diagnostics,
            global_r2=global_r2,
        )
        summary_rows.append(row)

        summary_bits = [
            f"{name}_R2={diagnostics[name]['R2']:.4f} local={diagnostics[name]['local_R2']:.4f} vs mean={diagnostics[name]['reference_mean_fraction']:.4f}"
            for name in args.species
        ]
        progress("  " + ", ".join(summary_bits) + f", total={global_r2:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("curtain_datetime").reset_index(drop=True)
    summary_df.to_csv(args.summary_csv, index=False)
    progress(f"Saved CSV summary to {args.summary_csv}")

    plot_r2_vs_date(
        summary_df=summary_df,
        active_species=args.species,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
    )
    progress(f"Saved figure to {args.output_png}")
    progress(f"Saved figure to {args.output_pdf}")


if __name__ == "__main__":
    main()
