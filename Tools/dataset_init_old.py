import os
import re
import glob
import sys
import dask
from dask.distributed import Client, LocalCluster
import xarray as xr
import Constants as cst
from datetime import datetime
if __name__ == '__main__':
    sys.path.append('..')
import File_management.json_file_editer as JFE
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Calculation.job_manager import JobManager as JM  # alias juste pour les types
from Log_management import Log


###############################################################################
# 1. Dask client creation
###############################################################################

def get_dask_client() -> Client:
    """Set up a Dask Client."""
    dask.config.set({'distributed.dashboard.link': 'http://localhost:{port}/status'})
    #cluster = LocalCluster(dashboard_address=':45451')
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        dashboard_address=':45451'
    )
    Log.add_to_log(f"The cluster :\n {cluster}")
    client = Client(cluster)
    return client

###############################################################################
# 2. Single-file approach (unchanged from your snippet)
###############################################################################


def normalize_height_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensures 'height' is length 90 and 'height_2' is length 91.
    Also renames 'height_bnds' / 'height_2_bnds' accordingly.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to normalize.

    Returns
    -------
    xr.Dataset
        Dataset with consistent 'height' (length=90) and 'height_2' (length=91).
    """
    # If 'height_2' is 90, we want it to be 'height'
    if "height_2" in ds.dims and len(ds["height_2"]) == 90:
        ds = ds.rename_dims({"height_2": "height_temp"}).rename_vars({"height_2": "height_temp"})
        if "height_2_bnds" in ds.variables:
            ds = ds.rename_vars({"height_2_bnds": "height_bnds_temp"})

    # If 'height' is 91, we want it to be 'height_2'
    if "height" in ds.dims and len(ds["height"]) == 91:
        ds = ds.rename_dims({"height": "height_2_temp"}).rename_vars({"height": "height_2_temp"})
        if "height_bnds" in ds.variables:
            ds = ds.rename_vars({"height_bnds": "height_2_bnds_temp"})


    if "height_temp" in ds.dims :
        ds = ds.rename_dims({"height_temp": "height"}).rename_vars({"height_temp": "height"})
        if "height_bnds_temp" in ds.variables:
            ds = ds.rename_vars({"height_bnds_temp": "height_bnds"})
    if "height_2_temp" in ds.dims :
        ds = ds.rename_dims({"height_2_temp": "height_2"}).rename_vars({"height_2_temp": "height_2"})
        if "height_2_bnds_temp" in ds.variables:
            ds = ds.rename_vars({"height_2_bnds_temp": "height_2_bnds"})

    # Unify the 'bounds' attribute for height and height_2 to avoid conflicts
    if "height" in ds.coords and "height_bnds" in ds.variables:
        ds["height"].attrs["bounds"] = "height_bnds"
    if "height_2" in ds.coords and "height_2_bnds" in ds.variables:
        ds["height_2"].attrs["bounds"] = "height_2_bnds"

    #print(ds.dims)


    #print(ds.variables)
    return ds


def add_dzghalf(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes dzghalf (difference in geopotential height between adjacent levels)
    from zghalf and adds it as a new variable in the dataset.
    """
    Log.add_to_log("(DI0001) Adding dzghalf to the dataset...")
    if 'zghalf' not in ds:
        Log.add_to_log("Variable 'zghalf' not found in the dataset. Skipping dzghalf computation.")
        return ds


    # 1) Compute the difference along 'height_2'. By default, 'diff' reduces the size of that dimension by 1.
    #    e.g., if 'height_2' has length N, the new array shape will have length N-1 along that dimension.
    dz = ds['zghalf'].diff(dim='height_2', label='lower').rename(height_2="height").drop_vars("height")

    # 2) Assign the new DataArray to the Dataset. This approach keeps the dimension
    #    and coordinate information consistent automatically.
    dzghalf = -1 * dz
    #ds = ds.assign(dzghalf=-dz) # could also be ds['dzghalf'] = -dz
    # Persist the dataset in memory (if it's Dask-backed)
    """if hasattr(dzghalf.data, 'persist'):  # Check if it's a Dask array
        Log.add_to_log("(DI0011) Persisting dzghalf in memory...")
        dzghalf = dzghalf.persist()"""
    ds['dzghalf'] = dzghalf
    # 3) Optionally rename the coordinate or adjust attributes:
    #ds['dzghalf'].attrs['long_name'] = "Layer thickness in geopotential height"
    #ds['dzghalf'].attrs['units'] = "m"  # or whatever units are appropriate
    Log.add_to_log("(DI0002) dzghalf was added to the dataset...")

    return ds


def clip_wa(ds):
    """
    Using wa directly is problematic. wa has 91 level instead of 90. Here, we get rid of the highest level, then
    force a specific height, number of cells and time.
    :param ds: the dataset as an xarray.Dataset
    :return:
    """
    #Log.add_to_log(f"The dataset before wa is \n{ds}")
    wa = xr.DataArray(
        data=ds["wa"].sel(height_2=slice(2, None)),
        dims=("time", "height", "ncells"),
        coords={ "time": ds.time, "height": ds.height, "ncells": ds.ncells},
    )
    return ds.assign(wa=wa)

def open_dataset_from_pattern(
        file_pattern: str,
        var_used=None,
        date_fr=None,
        date_to=None,
        chunks="auto"
) -> xr.Dataset:
    """
    Opens a single NetCDF file or multiple files. Ensures:
      - 'height' always has length=90
      - 'height_2' always has length=91

    If multiple files are found, each is preprocessed individually,
    then combined into a final Dataset.

    Parameters
    ----------
    file_pattern : str
        Path or pattern to NetCDF files, e.g. "*.nc".
    var_used : list or None
        Optionally select only these variables from the dataset.
    date_fr : str or None
        Start date for time slicing (e.g. "1990-01-01").
    date_to : str or None
        End date for time slicing (e.g. "1990-02-01").
    chunks : dict or str
        Chunking specification for dask.

    Returns
    -------
    xr.Dataset
        Combined Dataset with consistent 'height' and 'height_2'.
    """
    # 1) Resolve file paths from the pattern
    if os.path.isfile(file_pattern):
        file_paths = [file_pattern]
    else:
        file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        raise FileNotFoundError(f"No files found matching the pattern: {file_pattern}")

    # 2) Open and normalize each file individually
    datasets = []
    for path in file_paths:
        ds_single = xr.open_dataset(path, chunks=chunks, engine="h5netcdf")
        ds_single = normalize_height_dims(ds_single)  # Fix height vs. height_2
        # Optionally select variables
        if var_used is not None:
            ds_single = ds_single[var_used]
        
        ds_single = add_dzghalf(ds_single)

        datasets.append(ds_single)

    # 3) Combine into a single Dataset
    if len(datasets) == 1:
        ds_combined = datasets[0]
    else:
        # Combine by coordinates, or use xr.concat as needed
        ds_combined = xr.combine_by_coords(datasets)
    if 'wa' in ds_combined:
        Log.add_to_log("(DI0003) Variable 'wa' is in the dataset. Changing its dimensions for a height level of 90 instead of 91.")
        ds_combined = ds_combined.pipe(clip_wa)

    # Perform chunking at the very end
    ds_combined = ds_combined.chunk(chunks)
    ds_combined = ds_combined.sel(time=slice(date_fr, date_to))


    return ds_combined


###############################################################################
# 3. Multi-file approach: open_mfdataset + date-filter from filenames
###############################################################################


def _parse_date_from_filename(fname: str) -> datetime:
    """
    Extracts a date from filenames like:
      mbe3056_atm_3d_POD1_ml_19900101T000000Z.nc
    Adjust the regex if your filenames differ.
    """
    match = re.search(r"_([0-9]{8}T[0-9]{6})Z", fname)
    if match:
        date_str = match.group(1)  # e.g. '19900101T000000'
        return datetime.strptime(date_str, "%Y%m%dT%H%M%S")
    return None


def make_ds_with_PO_data(data_dir: str,
        date_fr: datetime,
        date_to: datetime,
        var_used=None,
        chunks='auto',
        file_pattern_2d: str = "*.nc",
        file_pattern_3d: str = "*.nc",
        ) -> xr.Dataset:
    """
    Creates a dataset from P-O data files in a specified directory.
    This function is specifically designed for scheme_comparison project data.
    
    Key differences from ds_from_directory:
    - Loads vertical grid data from pod0001_atm_vgrid_ml.nc (contains dzghalf)
    - Merges grid data with time-varying data
    - No need to compute dzghalf as it's already in the grid file

    Parameters
    ----------
    data_dir : str
        Directory containing the P-O data files.
    date_fr : datetime
        Start date for filtering files.
    date_to : datetime
        End date for filtering files.
    var_used : list, optional
        List of variables to include in the dataset.
    chunks : str or dict, optional
        Chunking strategy for Dask.
    file_pattern : str, optional
        File pattern to match P-O data files.

    Returns
    -------
    xr.Dataset
        Combined dataset from the P-O data files with grid data included.
    """
    Log.add_to_log(f"(DI0020) Starting PO dataset creation from {data_dir}")
    
    # Gather matching time-varying files
    all_files_2d = sorted(glob.glob(os.path.join(data_dir, file_pattern_2d)))
    all_files_3d = sorted(glob.glob(os.path.join(data_dir, file_pattern_3d)))
    # For 2D files, do not filter by filename date
    selected_files_2d = all_files_2d
    # For 3D files, filter by filename date as before
    selected_files_3d = []
    for f in all_files_3d:
        base = os.path.basename(f)
        # Skip the vertical grid file as we'll handle it separately
        if "vgrid" in base:
            continue
        dt = _parse_date_from_filename(base)
        if dt is not None and (dt >= date_fr and dt <= date_to):
            selected_files_3d.append(f)
    selected_files = selected_files_2d + selected_files_3d

    Log.add_to_log(f"(DI0021) Found {len(selected_files)} time-varying files in date range")

    if not selected_files:
        raise ValueError("No matching time-varying files found in the given date range.")

    # Load the vertical grid file (contains dzghalf and other grid variables)
    vgrid_file = os.path.join(data_dir, "pod0001_atm_vgrid_ml.nc")
    
    ds_vgrid = None
    if os.path.exists(vgrid_file):
        Log.add_to_log(f"(DI0022) Loading vertical grid data from {vgrid_file}")
        ds_vgrid = xr.open_dataset(vgrid_file, chunks=chunks, engine="h5netcdf")
        Log.add_to_log(f"(DI0023) Grid variables available: {list(ds_vgrid.variables.keys())}")
        # Log the dimensions and their sizes
        Log.add_to_log(f"(DI0024) Dimensions in vgrid: {ds_vgrid.dims}")
        if 'height_2' in ds_vgrid['dzghalf'].dims:
            ds_vgrid['dzghalf'] = ds_vgrid['dzghalf'].rename({'height_2': 'height'})
    else:
        Log.add_to_log(f"(DI0024) Warning: Vertical grid file not found at {vgrid_file}")

    ds_hgrid = None
    if var_used and ('cell_area' in var_used or 'clon' in var_used or 'clat' in var_used):
        # Load the horizontal grid file (contains cell areas for R02B09)
        # Standard ICON grid file location for R02B09, 0055 grid
        hgrid_file = cst.path_to_r02b09_gridfile
        if os.path.exists(hgrid_file):
            Log.add_to_log(f"(DI0037) Loading horizontal grid data from {hgrid_file}")
            try:
                ds_hgrid = xr.open_dataset(hgrid_file, chunks=chunks, engine="h5netcdf")
                # Look for cell area variable (based on R02B09 grid file structure)
                if 'cell_area' in ds_hgrid.variables:
                    Log.add_to_log(f"(DI0039) Successfully loaded cell areas from {hgrid_file}")
                else:
                    Log.add_to_log(f"(DI0040) No cell area variable found in {hgrid_file}")
                    ds_hgrid = None
            except Exception as e:
                Log.add_to_log(f"(DI0041) Could not load {hgrid_file}: {e}")
                ds_hgrid = None
        
        if ds_hgrid is None:
            Log.add_to_log("(DI0042) Warning: No horizontal grid file with cell areas found")
            Log.add_to_log(f"(DI0043) Searched standard location: {cst.path_to_r02b09_gridfile}")

    # Process time-varying files
    datasets = []
    for fpath in selected_files:
        ds = xr.open_dataset(fpath, chunks=chunks)
        # Select time range if time is a dimension
        if 'time' in ds.dims:
            ds = ds.sel(time=slice(date_fr, date_to))
            # Only add if time dimension is not zero
            if ds.sizes['time'] > 0:
                datasets.append(ds)
        else:
            datasets.append(ds)

    # Combine time-varying datasets
    Log.add_to_log(f"(DI0019) Combining {len(datasets)} time-varying datasets")
    if len(datasets) == 1:
        ds_combined = datasets[0]
    else:
        ds_combined = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")

    # Merge with vertical grid data if available
    if ds_vgrid is not None:
        Log.add_to_log("(DI0018) Merging time-varying data with vertical grid data")
        
        # Find common dimensions between the datasets
        common_dims = set(ds_combined.dims.keys()) & set(ds_vgrid.dims.keys())
        Log.add_to_log(f"(DI0017) Common dimensions: {common_dims}")
        
        # Merge the datasets - grid variables will be broadcast across time
        try:
            ds_combined = xr.merge([ds_combined, ds_vgrid], combine_attrs="drop_conflicts", compat='override')
            Log.add_to_log("(DI0016) Successfully merged with vertical grid data")
        except Exception as e:
            Log.add_to_log(f"(DI0015) Warning: Could not merge grid data: {e}")
            Log.add_to_log("(DI0014) Continuing with time-varying data only")

    # Merge with horizontal grid data (cell areas) if available
    if ds_hgrid is not None:
        Log.add_to_log("(DI0044) Merging with horizontal grid data (cell areas)")
        # Find common dimensions
        if 'ncells' in ds_combined.dims and 'cell' in ds_hgrid.dims:
            if ds_combined.dims['ncells'] == ds_hgrid.dims['cell']:
                # Rename 'cell' to 'ncells' to match
                ds_hgrid = ds_hgrid.rename({'cell': 'ncells'})
                Log.add_to_log(f"(DI0049) Renamed 'cell' dimension to 'ncells'")
                area_dims = ds_hgrid['cell_area'].dims
                common_dims_h = set(area_dims) & set(ds_combined.dims.keys())
                Log.add_to_log(f"(DI0050) After renaming - common dimensions: {common_dims}")
            else:
                Log.add_to_log(f"(DI0051) Dimension size mismatch: ncells={ds_combined.dims['ncells']}, cell={ds_hgrid.dims['cell']}")
                raise ValueError("Dimension size mismatch between time-varying data and horizontal grid data.")
        
        
        try:
            # Only merge the cell_area variable to avoid dimension conflicts
            if 'cell_area' in ds_hgrid.variables:
                cell_area_data = ds_hgrid[['cell_area']]
                ds_combined = xr.merge([ds_combined, cell_area_data], combine_attrs="drop_conflicts")
                Log.add_to_log("(DI0046) Successfully merged cell area data")
            else:
                Log.add_to_log("(DI0047) No cell_area variable found in horizontal grid")
        except Exception as e:
            Log.add_to_log(f"(DI0048) Warning: Could not merge horizontal grid data: {e}")
            Log.add_to_log("(DI0049) Continuing without cell area information")

    # Filter variables if specified
    if var_used is not None:
        # Filter out variables that don't exist in the dataset
        available_vars = [var for var in var_used if var in ds_combined.variables]
        missing_vars = [var for var in var_used if var not in ds_combined.variables]
        
        if missing_vars:
            Log.add_to_log(f"(DI0032) Warning: Variables not found in dataset: {missing_vars}")
        
        if available_vars:
            ds_combined = ds_combined[available_vars]
            Log.add_to_log(f"(DI0033) Selected variables: {available_vars}")
        else:
            Log.add_to_log("(DI0034) Warning: No requested variables found in dataset")

    # After combining, select the time range from the data (not filenames)
    ds_combined = ds_combined.sel(time=slice(date_fr, date_to))

    Log.add_to_log(f"(DI0035) Final dataset shape: {dict(ds_combined.dims)}")
    Log.add_to_log(f"(DI0036) Final dataset variables: {list(ds_combined.variables.keys())}")
    
    return ds_combined

def ds_from_directory(
        data_dir: str,
        date_fr: datetime,
        date_to: datetime,
        var_used=None,
        chunks='auto',
        file_pattern: str = "*.nc"
        ) -> xr.Dataset:
    """
    1. Initializes a Dask client.
    2. Gathers matching .nc files from `data_dir` using `file_pattern`.
    3. Filters files by date range (extracted from filename).
    4. Opens each file, normalizes its dimensions and attributes,
       and (optionally) selects specific variables.
    5. Combines the preprocessed datasets.
    6. Optionally applies a time slice.

    Returns:
      A combined xarray.Dataset with consistent 'height' and 'height_2'.
    """
    # Initialize Dask client (assume get_dask_client is defined elsewhere)
    # after numeral tests, it was found that the dask client does not help here
    #client = get_dask_client()
    #Log.add_to_log(f"(DI0004) Dask client initialized: {client}")
    Log.add_to_log(f"(DI0005) Starting dataset from directory")

    # Gather candidate files from the data directory
    all_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    selected_files = []
    for f in all_files:
        base = os.path.basename(f)
        dt = _parse_date_from_filename(base)
        if dt is not None and (dt >= date_fr and dt <= date_to):
            selected_files.append(f)

    Log.add_to_log(f"(DI0006) Found {len(selected_files)} files in {data_dir} from {date_fr} to {date_to}.")

    if len(selected_files) == 0:
        raise ValueError("No matching files found in the given date range.")

    # Process each file individually
    datasets = []
    for fpath in selected_files:
        Log.add_to_log(f"(DI0012) Opening file {fpath}")
        ds_single = xr.open_dataset(fpath, chunks=chunks, engine="h5netcdf")
        #ds_single = xr.open_dataset(fpath, chunks=chunks, engine="h5netcdf")
        # verify that height and height_2 are in the right order
        ds_single = normalize_height_dims(ds_single)

        # Compute and add dzghalf if zghalf exists
        if 'zghalf' in ds_single.variables and 'dzghalf' not in ds_single.variables:
            ds_single = add_dzghalf(ds_single)
        if 'wa' in ds_single.variables:
            ds_single = ds_single.pipe(clip_wa)
        ds_single = ds_single.sel(time=slice(date_fr, date_to))

        datasets.append(ds_single)

    # Merge the datasets using combine_by_coords to respect coordinates along shared dimensions
    if len(datasets) == 1:
        ds_combined = datasets[0]
    else:
        ds_combined = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
    if var_used is not None:
        ds_combined = ds_combined[var_used]
    return ds_combined


###############################################################################
# 4. Original NextGEMS Catalog approach
###############################################################################
try:
    import intake
except ImportError:
    intake = None
    Log.add_to_log("WARNING: intake is not installed; ds_from_catalog will fail.")

def ds_from_catalog(
    var_used,
    date_fr,
    date_to,
    model,
    project,
    sim_ID,
    zoom=8,
    time_resolution="P1D",
    chunks="auto"
) -> xr.Dataset:
    """
    Original approach using the NextGEMS intake catalog.
    """
    Log.add_to_log("(DI0007) Initialising DS from catalog")
    client = get_dask_client()
    Log.add_to_log(f"(DI0008) Dask client done, client is {client}")

    if not intake:
        raise ImportError("intake is not installed; cannot open NextGEMS catalog.")

    catalog_url = "https://data.nextgems-h2020.eu/catalog.yaml"
    cat = intake.open_catalog(catalog_url)

    # If sim_ID is given, we open that node specifically
    if sim_ID is not None:
        ds = cat[model][project][sim_ID](
            xarray_kwargs={"engine": "h5netcdf"},
            zoom=zoom,
            time=time_resolution,
            chunks=chunks
        ).to_dask()
    else:
        ds = cat[model][project](
            xarray_kwargs={"engine": "h5netcdf"},
            zoom=zoom,
            time=time_resolution,
            chunks=chunks
        ).to_dask()

    ds = ds[var_used].sel(time=slice(date_fr, date_to))
    Log.add_to_log("(DI0009) Initialising from catalog done")
    return ds

###############################################################################
# 5. ds_init: orchestrates loading from P-O, Jakob, or Catalog
###############################################################################
def ds_init_2d(myJob: 'JM', var_names, chunks='auto') -> None:
    pass
    return

def ds_init_3d(myJob: 'JM', chunks='auto') -> xr.Dataset:
    """
    Dispatches to either:
      - ds_from_directory() for P-O or Jakob multiple-file approach
      - ds_from_abspath() for single-file testing
      - ds_from_catalog() for NextGEMS usage
    based on the 'project' field in 'options'.
    """
    date_fr = myJob.start_datetime  # e.g. '1990-01-01' in datetime format
    date_to = myJob.end_datetime  # e.g. '1990-01-27' in datetime format

    model = myJob.model
    project = myJob.project
    exp_name = myJob.exp_name # e.g. 'mbe3056' or 'jed0011' or 'pod0001'

    # Example directory/pattern for P-O
    # (You may need to refine the file_pattern to match your many .nc files)
    if 'mbe' in exp_name:
        DIR = f"/work/mh0066/m301130/ICON/icon-mpim-master_aea93e2c/experiments/{exp_name}"
        PATTERN = "mbe3056_atm_3d_*.nc"  # or something like "mbe3056_atm_2d_*1990*.nc"
        # there is a problem with 2d data
    elif 'jed' in exp_name:
        DIR = f"/work/bm1183/m301049/icon-mpim/experiments/{exp_name}"
        PATTERN = "jed0011_atm_3d_cloud_*.nc"
    elif 'pod' in exp_name:
    #     DIR = f"/work/mh0066/m301130/projects/{project}/{model}/build/experiments/{exp_name}"
        # DIR = f"/work/mh0066/m301130/projects/{project}/{model}/build_16-01-2025/experiments/{exp_name}"
        DIR = f"/work/mh0066/m301130/projects/{project}/{model}_19-01-2026/build_16-01-2025/experiments/{exp_name}"
        # /work/mh0066/m301130/projects/scheme_comparison/icon-mpim_19-01-2026/build_16-01-2025/experiments/pod0001
        PATTERN_3d = f"pod*_atm_3d_{myJob.time_interval_str}_*.nc"
        PATTERN_2d = f"pod*_atm_2d_{myJob.time_interval_str}_*.nc"  # to avoid pod0001_atm_vgrid_ml_*.nc

    # Example directory/pattern for Jakob
    # (Adjust to your actual multi-file pattern)

    # Decide which approach to use:
    if project == 'scheme_comparison':
        Log.add_to_log("(DI0010) making data from P-O files requires small differences in the code, hardly automisable.")
        ds = make_ds_with_PO_data(
            data_dir=DIR,
            file_pattern_2d=PATTERN_2d,
            file_pattern_3d=PATTERN_3d,
            date_fr=date_fr,
            date_to=date_to,
            var_used=myJob.var_names,
            chunks=chunks
        )
    elif project == 'Jakob' or ('mbe' in exp_name):
        Log.add_to_log("(DI0011) Loading multiple P-O files from directory...")
        ds = ds_from_directory(
            data_dir=DIR,
            file_pattern=PATTERN,
            date_fr=date_fr,
            date_to=date_to,
            var_used=myJob.var_names,
            chunks=chunks
        )

    elif project == 'catalog':
        # For a NextGEMS-like project using Intake
        ds = ds_from_catalog(
            var_used=myJob.var_names,
            date_fr=date_fr,
            date_to=date_to,
            model=model,
            project=project,
            sim_ID=simulation_ID,
            chunks=chunks
        )

    # Log detailed dataset information
    log_dataset_info(ds, f"Final Dataset for {project}")
    
    return ds

###############################################################################
# 6. Example main usage / tests
###############################################################################
if __name__ == '__main__':
    Log.init_log('/work/mh0066/m301130/MP_Analysis_Slave/Jobs/Job_instant_Earth_PO')
    ###########################################################################
    # Example A: Single-File Test
    ###########################################################################
    single_file_path = '/work/bm1183/m301049/icon-mpim/experiments/jed0011/jed0011_atm_3d_cloud_19790630T000040Z.15356915.nc'
    print("start ds Jakob,")
    ds_a = open_dataset_from_pattern(
        file_pattern=single_file_path,
        var_used=None,         # or ['clc', 't', etc.]
        date_fr='1979-06-30',  # Only partial time slicing if file has those coords
        date_to='1979-07-01',
        chunks={"time": 1, "height": 10}
    )

    print("Single-file Jakob:", ds_a)

    '''print("start ds POD1,")
    single_file_path = '/work/mh0066/m301130/ICON/icon-mpim-master_aea93e2c/experiments/mbe3056/mbe3056_atm_3d_POD1_ml_19900102T000000Z.nc'
    ds_a = open_dataset_from_pattern(
        file_pattern=single_file_path,
        var_used=None,  # or ['clc', 't', etc.]
        date_fr=None,  # Only partial time slicing if file has those coords
        date_to=None,
        chunks='auto'
    )
    print("Single-file PO1:", ds_a)

    print("start ds POD2,")
    single_file_path = '/work/mh0066/m301130/ICON/icon-mpim-master_aea93e2c/experiments/mbe3056/mbe3056_atm_3d_POD2_ml_19900102T000000Z.nc'
    ds_a = open_dataset_from_pattern(
        file_pattern=single_file_path,
        var_used=None,  # or ['clc', 't', etc.]
        date_fr=None,  # Only partial time slicing if file has those coords
        date_to=None,
        chunks='auto'
    )
    print("Single-file PO2:", ds_a)
    print('This is rho')
    print(ds_a["rho"].sel(height=90))

    print("start ds POD3,")
    single_file_path = '/work/mh0066/m301130/ICON/icon-mpim-master_aea93e2c/experiments/mbe3056/mbe3056_atm_3d_POD3_ml_19900102T000000Z.nc'
    ds_a = open_dataset_from_pattern(
        file_pattern=single_file_path,
        var_used=None,  # or ['clc', 't', etc.]
        date_fr=None,  # Only partial time slicing if file has those coords
        date_to=None,
        chunks='auto'
    )
    print("Single-file PO3:", ds_a)

    print("start ds POD4,")
    single_file_path = '/work/mh0066/m301130/ICON/icon-mpim-master_aea93e2c/experiments/mbe3056/mbe3056_atm_3d_POD4_ml_19900102T000000Z.nc'
    ds_a = open_dataset_from_pattern(
        file_pattern=single_file_path,
        var_used=None,  # or ['clc', 't', etc.]
        date_fr=None,  # Only partial time slicing if file has those coords
        date_to=None,
        chunks='auto'
    )
    print("Single-file PO4:", ds_a)'''

    print("Try patterns")
    print("start ds POD all!!,")
    single_file_path = '/work/mh0066/m301130/ICON/icon-mpim-master_aea93e2c/experiments/mbe3056/mbe3056_atm_3d_POD*_ml_19900102T000000Z.nc'
    ds_a = open_dataset_from_pattern(
        file_pattern=single_file_path,
        var_used=None,  # or ['clc', 't', etc.]
        date_fr=None,  # Only partial time slicing if file has those coords
        date_to=None,
        chunks='auto'
    )
    print("pattern POD all! :", ds_a)

    try_multiple = True
    if try_multiple:
        ###########################################################################
        # Example B: Loading multiple P-O files via ds_init
        ###########################################################################
        print('multiple files now :')
        job_folderpath = '/work/mh0066/m301130/MP_Analysis_Slave/Jobs/Job_instant_Earth_PO/options.json'
        option_json = JFE.get_json_data(job_folderpath)
        var_names_po = None  # or e.g. ['temperature','pressure']
        ds_b = ds_init(option_json, var_names_po, chunks="auto")
        print("Multiple-file P-O DS:", ds_b)

        ###########################################################################
        # Example C: Loading multiple Jakob files via ds_init
        ###########################################################################
        job_folderpath = '/work/mh0066/m301130/MP_Analysis_Slave/Jobs/Job_aqua_jakob/options.json'
        option_json = JFE.get_json_data(job_folderpath)
        var_names_jakob = None  # or e.g. ['t', 'rv']
        ds_c = ds_init(options_jakob, var_names_jakob, chunks="auto")
        print("Multiple-file Jakob DS:", ds_c)

        ###########################################################################
        # Example D: Loading from NextGEMS catalog
        ###########################################################################
        # This requires intake installed, and your model/project/sim_ID must match the catalog entries
        # options_catalog = {
        #     'parameters': {
        #         'date_fr': '2000-01-01',
        #         'date_to': '2000-01-05'
        #     },
        #     'description': {
        #         'model': 'ICON',
        #         'project': 'C5',
        #         'simulation_ID': 'AMIP_CNTL',
        #         'zoom': 8,
        #         'time': 'P1D'
        #     }
        # }
        # var_names_catalog = ['clw', 'cli']
        # ds_d = ds_init(options_catalog, var_names_catalog, chunks="auto")
        # print("Catalog-based DS:", ds_d)


###############################################################################
# 7. Additional utilities for grid cell areas
# mainly for tests
###############################################################################
def load_icon_grid_areas(grid_file_path: str, chunks='auto') -> xr.Dataset:
    """
    Load ICON grid file and extract cell areas.
    
    Parameters
    ----------
    grid_file_path : str
        Path to the ICON grid file (e.g., R02B09 grid file)
    chunks : str or dict, optional
        Chunking strategy for Dask
        
    Returns
    -------
    xr.Dataset
        Dataset containing only the cell_area variable
    """
    try:
        Log.add_to_log(f"(DI0050) Loading ICON grid from {grid_file_path}")
        ds_grid = xr.open_dataset(grid_file_path, chunks=chunks, engine="h5netcdf")
        
        # Look for cell area variable (based on actual R02B09 grid file content)
        # Primary: cell_area, cell_area_p (both are "area of grid cell" in m²)
        # Alternative: dual_area, dual_area_p (areas of dual hexagonal/pentagonal cells)
        area_vars = ['cell_area', 'cell_area_p', 'dual_area', 'dual_area_p']
        
        for area_var in area_vars:
            if area_var in ds_grid.variables:
                Log.add_to_log(f"(DI0051) Found cell area variable: {area_var}")
                # Create dataset with only the area variable, renamed to standard name
                area_data = ds_grid[[area_var]].rename({area_var: 'cell_area'})
                Log.add_to_log(f"(DI0052) Cell area shape: {area_data.cell_area.shape}")
                return area_data
        
        Log.add_to_log(f"(DI0053) No recognized cell area variable found in {grid_file_path}")
        Log.add_to_log(f"(DI0054) Available variables: {list(ds_grid.variables.keys())}")
        return None
        
    except Exception as e:
        Log.add_to_log(f"(DI0055) Error loading grid file {grid_file_path}: {e}")
        return None

def add_cell_areas_to_dataset(ds: xr.Dataset, grid_file_path: str, chunks='auto') -> xr.Dataset:
    """
    Add cell areas from an ICON grid file to an existing dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to add cell areas to
    grid_file_path : str
        Path to the ICON grid file containing cell areas
    chunks : str or dict, optional
        Chunking strategy for Dask
        
    Returns
    -------
    xr.Dataset
        Dataset with cell_area variable added
    """
    area_data = load_icon_grid_areas(grid_file_path, chunks=chunks)
    
    if area_data is not None:
        try:
            # Merge the area data with the existing dataset
            ds_with_areas = xr.merge([ds, area_data], combine_attrs="drop_conflicts")
            Log.add_to_log("(DI0056) Successfully added cell areas to dataset")
            return ds_with_areas
        except Exception as e:
            Log.add_to_log(f"(DI0057) Error merging cell areas: {e}")
            return ds
    else:
        Log.add_to_log("(DI0058) Could not load cell areas, returning original dataset")
        return ds

def test_r02b09_grid_access():
    """
    Test function to check if the R02B09 grid file can be accessed and loaded.
    Useful for debugging grid file availability.
    
    Returns
    -------
    bool
        True if grid file is accessible and contains cell areas, False otherwise
    """
    grid_path = "/pool/data/ICON/grids/public/mpim/0055/icon_grid_0055_R02B09_G.nc"
    
    Log.add_to_log(f"(DI0059) Testing access to R02B09 grid file: {grid_path}")
    
    if not os.path.exists(grid_path):
        Log.add_to_log("(DI0060) ERROR: Grid file does not exist")
        return False
    
    try:
        Log.add_to_log("(DI0061) Grid file exists, attempting to open...")
        ds_test = xr.open_dataset(grid_path, engine="h5netcdf")
        
        Log.add_to_log(f"(DI0062) Grid file opened successfully")
        Log.add_to_log(f"(DI0063) Grid dimensions: {dict(ds_test.dims)}")
        Log.add_to_log(f"(DI0064) Available variables: {list(ds_test.variables.keys())}")
        
        # Check for cell area variables (based on actual R02B09 content)
        area_vars = ['cell_area', 'cell_area_p', 'dual_area', 'dual_area_p']
        found_areas = [var for var in area_vars if var in ds_test.variables]
        
        if found_areas:
            Log.add_to_log(f"(DI0065) SUCCESS: Found cell area variables: {found_areas}")
            for var in found_areas:
                Log.add_to_log(f"(DI0066) {var} shape: {ds_test[var].shape}")
                # Show units and description if available
                if hasattr(ds_test[var], 'units'):
                    Log.add_to_log(f"(DI0067) {var} units: {ds_test[var].units}")
                if hasattr(ds_test[var], 'long_name'):
                    Log.add_to_log(f"(DI0068) {var} description: {ds_test[var].long_name}")
            return True
        else:
            Log.add_to_log("(DI0069) WARNING: No recognized cell area variables found")
            return False
            
    except Exception as e:
        Log.add_to_log(f"(DI0068) ERROR: Could not open grid file: {e}")
        return False

def log_dataset_info(ds: xr.Dataset, context: str = "Dataset"):
    """
    Log detailed information about a dataset's variables, dimensions, and structure.
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to analyze
    context : str
        Context description for the logs
    """
    Log.add_to_log(f"(DI0100) ========== {context} Information ==========")
    
    # Dataset dimensions
    Log.add_to_log(f"(DI0101) Dimensions: {dict(ds.dims)}")
    
    # Coordinates
    if ds.coords:
        Log.add_to_log(f"(DI0102) Coordinates:")
        for coord_name, coord in ds.coords.items():
            Log.add_to_log(f"  {coord_name}: {coord.shape} {coord.dtype}")
    
    # Data variables
    Log.add_to_log(f"(DI0103) Data Variables:")
    for var_name, var in ds.data_vars.items():
        dims_str = f"({', '.join(var.dims)})" if var.dims else "(scalar)"
        units = var.attrs.get('units', 'no units')
        long_name = var.attrs.get('long_name', 'no description')
        Log.add_to_log(f"  {var_name}: {var.shape} {var.dtype} {dims_str}")
        Log.add_to_log(f"    Units: {units}")
        Log.add_to_log(f"    Description: {long_name}")
    
    # Memory usage estimation
    total_size = ds.nbytes
    size_gb = total_size / (1024**3)
    Log.add_to_log(f"(DI0104) Estimated memory usage: {size_gb:.2f} GB")
    
    # Time range if time dimension exists
    if 'time' in ds.dims:
        time_coord = ds.coords.get('time', None)
        if time_coord is not None:
            try:
                start_time = str(time_coord.min().values)
                end_time = str(time_coord.max().values)
                n_times = len(time_coord)
                Log.add_to_log(f"(DI0105) Time range: {start_time} to {end_time} ({n_times} steps)")
            except Exception as e:
                Log.add_to_log(f"(DI0106) Could not determine time range: {e}")
    
    Log.add_to_log(f"(DI0107) ========== End {context} Information ==========")
