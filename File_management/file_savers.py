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
import threading
import fcntl

def save_to_netcdf(dfcopy, wa, frac_name, qmin, ta_start, ta_end, ta_nbpts, wa_pm, netcdf_filename=None, data_type='ta-wafixed'):
    """Save results to a shared NetCDF file with thread-safe access"""
    if data_type=='ta-wafixed':
        # Create filename based on parameters
        filename_ending = f"qmin{qmin}"
        
        netcdf_filename = f"MFD_{frac_name}_{filename_ending}.nc"
        
        # Convert DataFrame to xarray Dataset
        ds_result = xr.Dataset()
        
        # Create coordinate arrays
        unique_ta = sorted(dfcopy['ta'].unique())
        unique_frac = sorted(dfcopy[frac_name].unique())
        
        # Create mass_norm array
        pivot_table = dfcopy.pivot(index='ta', columns=frac_name, values='mass_norm')
        mass_norm_array = pivot_table.reindex(index=unique_ta, columns=unique_frac).values
        
        # Add data variables to dataset
        ds_result['mass_norm'] = (['ta', frac_name], mass_norm_array)
        ds_result['mass'] = (['ta', frac_name], 
                            dfcopy.pivot(index='ta', columns=frac_name, values='mass').reindex(index=unique_ta, columns=unique_frac).values)
        
        # Add coordinates
        ds_result.coords['ta'] = unique_ta
        ds_result.coords[frac_name] = unique_frac
        ds_result.coords['wa'] = wa
        
        # Add scalar parameters as attributes
        ds_result.attrs.update({
            'wa_pm': wa_pm,
            'ta_start': ta_start,
            'ta_end': ta_end,
            'ta_nbpts': ta_nbpts,
            'qmin': qmin,
            'frac_name': frac_name,
            'creation_time': pd.Timestamp.now().isoformat()
        })
        
        # Thread-safe file writing
        lock_filename = f"{netcdf_filename}.lock"
        
        with threading.Lock():
            try:
                # Try to open existing file and append
                if os.path.exists(netcdf_filename):
                    with open(lock_filename, 'w') as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                        
                        # Load existing data
                        ds_existing = xr.load_dataset(netcdf_filename)
                        
                        # Check if this wa value already exists
                        if 'wa_values' in ds_existing.dims and wa in ds_existing.coords.get('wa_values', []):
                            Log.add_to_log(f"   Warning: wa={wa} already exists in {netcdf_filename}, skipping...")
                            return
                        
                        # Expand dimensions to include multiple wa values
                        if 'wa_values' not in ds_existing.dims:
                            # First time adding multiple wa values
                            ds_existing = ds_existing.expand_dims('wa_values')
                            ds_existing.coords['wa_values'] = [ds_existing.attrs.get('wa', wa)]
                        
                        # Add new wa value
                        new_wa_values = list(ds_existing.coords['wa_values'].values) + [wa]
                        
                        # Expand current result to match existing structure
                        ds_result_expanded = ds_result.expand_dims('wa_values')
                        ds_result_expanded.coords['wa_values'] = [wa]
                        
                        # Concatenate along wa_values dimension
                        ds_combined = xr.concat([ds_existing, ds_result_expanded], dim='wa_values')
                        ds_combined.coords['wa_values'] = new_wa_values
                        
                        # Update global attributes
                        ds_combined.attrs.update(ds_result.attrs)
                        
                        # Save combined dataset
                        ds_combined.to_netcdf(netcdf_filename)
                        
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        
                else:
                    # Create new file
                    with open(lock_filename, 'w') as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                        
                        # Save as single wa value initially
                        ds_result.to_netcdf(netcdf_filename)
                        
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        
            except Exception as e:
                Log.add_to_log(f"   Error saving to NetCDF: {e}")
                # Fallback to CSV
                filename_data = f"MFD_{frac_name}_data_wa{wa}_{filename_ending}.csv"
                dfcopy.to_csv(f"./{filename_data}", index=False)
                Log.add_to_log(f"   Fallback: saved to CSV {filename_data}")
            finally:
                # Clean up lock file
                if os.path.exists(lock_filename):
                    os.remove(lock_filename)
    else:
        raise ValueError("Unsupported data_type. Use 'ta-wafixed'.")

