##########################
# use venv, not unstable
import intake
import dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import xarray as xr
'''xr.set_options(
        display_expand_data_vars=True,
        display_expand_coords=True,
        display_width=2000,
        display_max_rows=999999
    )'''
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xhistogram.xarray import histogram
#############
#############
from Tools import dataset_init
from Log_management import Log
from File_management import file_savers as FS
from File_management import json_file_editer as JFE
from Calculation import job_manager
import threading
import fcntl



def main():
    ##################################################
    # given qc/(qc+qi), wa, compute the mass-weighted histogram of qc/(qc+qi) vs ta
    # frac_name choices = ['qc_qci', 'qg_qgs', 'qr_qrgs']
    frac_name = 'qc_qci'; qmin = 1e-12  # qmin ensures that we are in cloudy conditions
    wa = 0.0; wa_pm = 0.04
    Log.init_log(os.path.join('./output'), filename=f"Log_MFD_fn{frac_name}_w{wa}.txt")
    Log.add_to_log("(MFD0001) Log nicely initialized ")

    ########################################
    # Load options and dataset
    options_json = JFE.get_json_data('options.json')
    Log.add_to_log(f"(MFD0002) Loaded options from options.json")
    myJob = job_manager.JobManager(options_json)
    ds = dataset_init.ds_init_3d(myJob, chunks={})

    ########################################
    # Define parameters
    ta_start=225.75; ta_end=285.25; ta_nbpts=120  # 120 points from 225.75 to 285.25
    Log.add_to_log(f"(MFD0003) arguments: ta_start={ta_start}, ta_end={ta_end}, ta_nbpts={ta_nbpts}, wa={wa}")
    ta_edges = np.linspace(ta_start, ta_end, ta_nbpts)  # 76 bins from 235.75 to 273.75
    Log.add_to_log(f"(MFD0004) Temperature edges: {ta_edges}")
    qc_qci_edges = np.linspace(0.0, 1.0, 101)
    
    #########################################
    # Process data in chunks to manage memory usage
    start = time.time()  # Start timing for the entire wa processing
    # Define time chunks for processing
    time_chunks = pd.date_range(start=ds['time'].min().values, end=ds['time'].max().values, freq='3H')
    Log.add_to_log(f"(MFD0005) Processing {len(time_chunks)} time chunks")

    #########################################
    # Initialize list to store histogram results from all time chunks
    histogram_results = []
    # Loop over time chunks
    for i in range(len(time_chunks)):
        if len(time_chunks) == 1:
            Log.add_to_log(f"   Processing time chunk {i+1}/{len(time_chunks)}: {time_chunks[i]} ")
            # For single time step, select all data
            ds_chunk = ds
        else:
            if i == len(time_chunks) - 1:
                # Last chunk - select from current time to end
                Log.add_to_log(f"   Processing time chunk {i+1}/{len(time_chunks)}: {time_chunks[i]} to end")
                ds_chunk = ds.sel(time=slice(time_chunks[i], None))
            else:
                # Use exclusive end to avoid overlap - select up to but not including the next chunk start
                next_time = time_chunks[i+1]
                Log.add_to_log(f"   Processing time chunk {i+1}/{len(time_chunks)}: {time_chunks[i]} to {next_time} (exclusive)")
                # Get all times >= start and < end
                time_mask = (ds['time'] >= time_chunks[i]) & (ds['time'] < next_time)
                ds_chunk = ds.isel(time=time_mask)
        
        ###############################################
        # Apply masks and filters for this chunk
        mask = ((ds_chunk['wa'] >= (wa - wa_pm)) & 
                (ds_chunk['wa'] < (wa + wa_pm)) & 
                (ds_chunk['clw'] > qmin))

        start_chunk = time.time()
        ds_ta = ds_chunk['ta'].where(mask)
        ds_ta.name = 'ta'
        
        ###############################################
        # Calculate cloud fraction based on frac_name, make weights
        if frac_name == 'qc_qci':
            ds_cloud_frac = (ds_chunk['clw'] / (ds_chunk['clw'] + ds_chunk['cli'])).where(mask)
            weights = (ds_chunk['rho'] * ds_chunk['cell_area'] * ds_chunk['dzghalf'] * (ds_chunk['clw'] + ds_chunk['cli'])).where(mask)
        elif frac_name == 'qg_qgs':
            ds_cloud_frac = (ds_chunk['qg'] / (ds_chunk['qg'] + ds_chunk['qs'])).where(mask)
            weights = (ds_chunk['rho'] * ds_chunk['cell_area'] * ds_chunk['dzghalf'] * (ds_chunk['qg'] + ds_chunk['qs'])).where(mask)
        elif frac_name == 'qr_qrgs':
            ds_cloud_frac = (ds_chunk['qr'] / (ds_chunk['qr'] + ds_chunk['qg'] + ds_chunk['qs'])).where(mask)
            weights = (ds_chunk['rho'] * ds_chunk['cell_area'] * ds_chunk['dzghalf'] * (ds_chunk['qr'] + ds_chunk['qg'] + ds_chunk['qs'])).where(mask)
        else:
            raise ValueError(f"Unknown frac_name: {frac_name}")
            
        ds_cloud_frac.name = frac_name
        weights.name = 'mass'
        
        ###############################################
        # Compute histogram for this chunk
        h_chunk = histogram(ds_cloud_frac, ds_ta, bins=[qc_qci_edges, ta_edges], weights=weights)
        h_chunk = h_chunk.compute()
        
        # Store the result
        histogram_results.append(h_chunk)
        
        end_chunk = time.time()
        Log.add_to_log(f"      Chunk {i+1} processed in {end_chunk - start_chunk:.2f} seconds")
    
    # Combine all histogram results
    Log.add_to_log(f"   Combining {len(histogram_results)} histogram chunks...")
    h = sum(histogram_results)  # Sum all histograms together
    Log.add_to_log(f"   Combined histogram computed.")

    ###############################################
    # Convert histogram to DataFrame for saving and plotting
    ta_centers = 0.5 * (ta_edges[:-1] + ta_edges[1:])
    qc_centers = 0.5 * (qc_qci_edges[:-1] + qc_qci_edges[1:])

    d1, d0 = h.dims
    h2 = (
        h.assign_coords({d0: ta_centers, d1: qc_centers})
        .rename("mass")
    )

    df = h2.to_dataframe().reset_index()
    df = df.rename(columns={d0: "ta", d1: frac_name})

    ta_pm_arr = 0.5 * np.diff(ta_edges)
    qc_pm_arr = 0.5 * np.diff(qc_qci_edges)

    df["ta_pm"] = np.repeat(ta_pm_arr, len(qc_centers))
    df[f"{frac_name}_pm"] = np.tile(qc_pm_arr, len(ta_centers))

    df["wa"] = wa
    df["wa_pm"] = wa_pm

    dfcopy = df.copy()
    totals = dfcopy.groupby("ta")["mass"].transform("sum")
    dfcopy["mass_norm"] = dfcopy["mass"] / totals

    table = dfcopy.pivot(index="ta", columns=frac_name, values="mass_norm")

    #################################################
    # Make and save the plot
    Log.add_to_log(f"   Making plot...")
    plt.figure(figsize=(8, 6))
    im = plt.pcolormesh(table.index, table.columns, table.values.transpose(), shading="auto", cmap="Greys", vmax=0.3)
    plt.colorbar(im, label="mass_norm")
    if frac_name == 'qc_qci':
        plt.ylabel(r"$\frac{q_c}{q_c+q_i}$")
    elif frac_name == 'qg_qgs':
        plt.ylabel(r"$\frac{q_g}{q_g+q_{s}}$")
    elif frac_name == 'qr_qrgs':
        plt.ylabel(r"$\frac{q_r}{q_r+q_g + q_s}$")
    plt.xlabel(r"$T$")
    plt.title(f"Distribution normalisée (wa={wa})")
    plt.tight_layout()


    filename_ending = f"_qmin{qmin}"
    filename_plot = f"MFD_{frac_name}_plot_wa{wa}_{filename_ending}.png"
    plt.savefig(f"./{filename_plot}", dpi=300, bbox_inches="tight")
    plt.close()

    #################################################
    # Save to NetCDF instead of CSV
    filename_data = f"MFD_{frac_name}_{filename_ending}.nc"
    FS.save_to_netcdf(dfcopy, wa, frac_name, qmin, ta_start, ta_end, ta_nbpts, wa_pm, netcdf_filename=filename_data, data_type='ta-wafixed')
    Log.add_to_log(f"   saved data to NetCDF and plot to {filename_plot}")

    end = time.time()
    length = end - start
    Log.add_to_log(f"   Processed wa={wa} in {int(length // 60)} minutes and {length % 60:.2f} seconds!")

if __name__ == "__main__":
    main()