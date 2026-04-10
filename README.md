
####################
Folder Structure

- Tools/: Functions for compiling and loading data from NetCDF files.
- File_management/: Functions for saving results and editing JSON files.
- Calculation/: Contains Job_manager, which manages job options and acts as a central object for your analysis. I use this to facilitate the use of the options for the job you want to send to a Levante.
- Main scripts: The main analysis workflow is in main.py. I give a structured example.

####################
What the example program does  :
This program loads my atmospheric simulation pod0001, processes it according to a map (ta --- qc/qci), computes *mass-weighted* histograms, and generates plots and NetCDF output. It is designed to run on the Levante supercomputer using SLURM (sbatch). 

####################
HOW TO USE
1. Set Your Options
    Edit options.json to select the dates and variables you want to analyze:
    - Under description, set start and end to your desired date range.
    - Under parameters -> instructions, specify the variables you want in variables_2d and variables_3d.

2. Prepare Your Job
    Make sure your SLURM script (e.g., run_main.sh) is set up to run main.py.
    - Submit your job using, in the terminal :  sbatch run_main.sh


3. Data Access
    Data is located in /work/mh0066/m301130/projects/scheme_comparison/icon-mpim/build/experiments/pod0001
    The dates of the data are:

    - in 2D, goes from Jan 1 1979 to Feb 1 1979, 
    - In 3D, Jan 11 to Jan 15 1979. 
    
    You select the dates you want to work with in options.json -> description, start and end

    Available variables:
    - 2D every 30min: lon, lat, cell_area, pr, tas, clivi, cllvi, qgvi, qrvi, qsvi
    - 2D every 3H: clivi, cllvi, qv2m, pr, pr_rain, pr_ice, pr_snow, pr_grpl, pres_msl, pres_sfc, prls, prw, qgvi, qrvi, qsvi, rlds, rlus, rlut, rsds, rsdt, rsus, rsut, tas, tauu, tauv, ts, uas, vas
    - 3D every 3H: clw, cli, qr, qg, qs, hus, ta, wa, ua, va, pfull, rho, zg, zghalf, dzghalf

4. Running the Analysis
    The main script (set as an example for you to base your work on) will:
    - Load the selected data and variables.
    - Process the data in time chunks to manage memory.
    - Compute histograms and statistics.
    - Save results as NetCDF and PNG plot files in the output directory.

