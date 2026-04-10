
import pandas as pd
import numpy as np
import os
import sys
from Tools import dataset_init
from Log_management import Log as Log
import time
from datetime import datetime
import Constants as cst
from File_management import json_file_editer as JFE
import copy

class CompAxis:
    def __init__(self, axis_dict):
        self.name = axis_dict.get("name", None)
        self.from_ = axis_dict.get("from", None)
        self.to = axis_dict.get("to", None)
        self.pm = axis_dict.get("pm", None)
        self.chunksize = axis_dict.get("chunksize", None)
        self.nchunks = self.get_nchunks()

    def __repr__(self):
        return f"axis(name={self.name}, from_={self.from_}, to={self.to}, pm={self.pm}, chunksize={self.chunksize}, nchunks={self.nchunks})"

    def get_nchunks(self):
        if self.chunksize is None or self.chunksize <= 0:
            _nchunks = 1
        elif self.chunksize is not None:
            _nchunks = int(np.abs(self.to - self.from_) // (2*self.pm*self.chunksize)) + int((1 if (np.abs(self.to - self.from_) % (2*self.pm*self.chunksize) > 0) else 0))
        return _nchunks


class JobManager:
    def __init__(self, option_json, exp_name=None, job_folder=None, job_num=None):

        self.option_json = option_json

        # Getting the description of the simulation
        self.model = option_json["description"]['model']
        self.project = option_json["description"]['project']
        if exp_name is not None:
            self.exp_name = exp_name
        else:
            self.exp_name = option_json["description"]['exp_name']
        self.job_folder = job_folder
        if job_num is not None:
            self.job_number = job_num
        else:
            self.job_number = option_json["Header"].get("job_number", 0)

        # Save the description of the simulation
        self.grid = option_json["description"]['grid']
        self.zoom = option_json["description"]['zoom']

        self.DataBase_path = cst.path_to_compileddata

        self.region_str = option_json["description"]['region']
        self.start_date_str = option_json["description"]['start']
        self.end_date_str = option_json["description"]['end']


        self.observables = option_json["parameters"]["instructions"]['observables']
        self.axis1 = CompAxis(option_json["parameters"]['Axis1'])
        self.axis2 = CompAxis(option_json["parameters"]['Axis2'])
        self.axis3 = CompAxis(option_json["parameters"]['Axis3'])
        self.header = option_json["Header"]
        self.created_jobfolderpath_list = []

        self.ndim = None
        self._var_names = None

        # self.calculate()

    @property
    def ndim(self):
        """
        The getter of the number of dimensions.
        The number of dimensions excludes the observables as a dimension. If axis1 and axis2 are activated, then there
        are 2 dimensions
        """
        if self._ndim is None :
            if self.axis1.name != None and self.axis2.name != None and self.axis3.name != None:
                self._ndim = 3
            elif self.axis1.name != None and self.axis2.name != None:
                self._ndim = 2
            elif self.axis1.name != None:
                self._ndim = 1
            else:
                self._ndim = 0
            # if the axis{ndim} has no "chunksize" keys, then it should not be considered as a new dimention.
            if getattr(self, f'axis{self._ndim}').chunksize is None and self._ndim > 1:
                self._ndim -= 1
            Log.add_to_log(f"ndim is now set to {self._ndim}. (JM0001)")
        return self._ndim

    @ndim.setter
    def ndim(self, val):
        self._ndim = val
    '''
    def find_used_var_list(self):
        # alwways added variables
        used_var = ['clw', 'cli', 'qs', 'qg', 'qr']

        # add variables linked to the name of the axis
        available_var_names = self.available_var_names()
        if self.ndim >= 1:
            if self.axis1.name in available_var_names:
                used_var.append(self.axis1.name)
        if self.ndim >= 2:
            if self.axis2.name in available_var_names:
                used_var.append(self.axis2.name)
        if self.ndim >= 3:
            if self.axis3.name in available_var_names:
                used_var.append(self.axis3.name)

        # axis for quantities that have to be measured
        if "Mass_density" in self.observables:
            vars = ['rho', 'dzghalf']
            for var in vars:
                if var not in used_var:
                    used_var.append(var)
    '''

    def are_jobs_created(self):
        # Chunksize should either be None or -1 in order to be ready to be run.
        if len(self.created_jobfolderpath_list) > 0:
            Log.add_to_log("(JM0009) Jobs already created, no need to create them again.")
            return True
        Log.add_to_log("(JM0012) No jobs created yet, need to create them.")
        return False

    def is_job_ready_for_compute(self):
        # Chunksize should either be None or -1 in order to be ready to be run.
        if self.job_number is None or self.job_number <= 0:
            Log.add_to_log("(JM0010) Job number is not set or negative, no job to be started.")
            return False
        

        # To run, one must be on a slurm job
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return False
        Log.add_to_log("(JM0011) Running on a SLURM job, can start a job.")
        return True
    
    @property
    def job_slurm_id(self):
        """
        Returns the SLURM job ID if available, otherwise returns None.
        """
        return os.environ.get("SLURM_JOB_ID", None)

    @property
    def start_datetime(self):   # output in datetime format
        from datetime import datetime
        return datetime.strptime(self.start_date_str, "%Y-%m-%dT%H:%M:%S")  # Adjust format to match your string

    @property
    def end_datetime(self):  # output in datetime format
        from datetime import datetime
        return datetime.strptime(self.end_date_str, "%Y-%m-%dT%H:%M:%S")  # Adjust format to match your string

    @property
    def dim_of_data_to_compute(self):
        """
        Is this 3d ICON data or 2d?
        """
        if self.compute_type == 'hydrometeor mass fraction':
            return '3d'
        return "2d"

    def get_exp_job_folderpath(self):
        os.path.join(cst.path_to_Jobs, self.exp_name)
        foldername = f"{self.exp_name}/{self.axis1['name']}_{self.axis2['name']}_{self.axis3['name']}"
        return foldername

    def get_axis1_foldername(self):
        foldername = f"{self.axis1['name']}{self.axis1['from']:.3e}-{self.axis1['to']:.3e}_step{self.axis1['step']:.3e}"
        return foldername

    def get_axis2_foldername(self):
        foldername = f"{self.axis2['name']}{self.axis2['val']:.3e}_pm{self.axis2['err']:.3e}"
        return foldername

    def get_axis2_filename(self):
        filename = f"{self.get_axis2_foldername()}.csv"
        return filename

    def get_axis3_filename(self):
        filename = f"{self.axis3['name']}{self.axis3['val']:.3e}_pm{self.axis3['err']:.3e}.csv"
        return filename

    def get_var_names_for_compute(self):
        var2d = self.option_json["parameters"]["instructions"]['variables_2d']
        var3d = self.option_json["parameters"]["instructions"]['variables_3d']
        var_names = var2d + var3d
        return var_names

    @property
    def var_names(self):
        if self._var_names is None:
            self._var_names = self.get_var_names_for_compute()
        return self._var_names
    @var_names.setter
    def var_names(self, value):
        # expects a list of strings
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError("var_names must be a list of strings.")
        self._var_names = value

    @property
    def time_interval_str(self):
        iso_duration = self.option_json["description"].get("time_interval", "PT3H")
        return iso_duration

    def calculate_fall_speed(self, q_x, rho, N_x, var):
        if var in ["clw", "cli", "qr", "qg", "qs"]:
            a_x, b_x, c_x, d_x = 0, 0, 0, 0
            if var == "clw":
                a_x, b_x, c_x, d_x = 523.6, 3, 1.2e3, 2
            elif var == "cli":
                a_x, b_x, c_x, d_x = 0.15, 2.05, 12.5, 0.4
            elif var == "qr":
                a_x, b_x, c_x, d_x = 523.6, 3, 130, 0.5
            elif var == "qs":
                a_x, b_x, c_x, d_x = 0.159, 2, 11.72, 0.5
            elif var == "qg":
                a_x, b_x, c_x, d_x = 0.3, 3, 22.5, 0.4

            # Compute mean diameter
            D_mean = ((q_x / (rho * N_x * a_x)) ** (1 / b_x))
            # Compute fall speed
            v_x = c_x * (D_mean ** d_x)
        else:
            v_x = 0
        return v_x

    def job_status_update(self, status):
        """
        Update the job status in the options.json file.
        """
        if self.job_folder is None:
            Log.add_to_log("(JM0013) Job folder is not set, cannot update job status.")
            return
        options_path = os.path.join(self.job_folder, cst.options_json_filename)
        if not os.path.exists(options_path):
            Log.add_to_log(f"(JM0014) Options file {options_path} does not exist, cannot update job status.")
            return
        options = JFE.get_json_data(options_path)
        options["Header"]["status"] = status
        JFE.dump_json_data(options, options_path)

    def create_jobs(self):
        """
        This program should be inside of a project. It does :
        1. Creates a folder inside of ../Jobs/ with name self.exp_name if it does not exist
        2. Inside of it, creates one folder (if it does not exist) with name "{axis1.name}_{axis2.name}_{axis3.name}" where axis1, axis2, and axis3 are the values 
           of the axes. Of course, the number of axis in the job folder name will depend on the number of dimensions (self.ndim) being used.
        3. Inside of it, creates multiple folders with names "jobi", with i being the ith job (of course, the ith job should not already exist, 
           and if it is the case, we should use a number not yet taken). The number of jobs to be made is equal to the multiplication of the number of chunks in the axis1, axis2, and 
           axis3 axes. 
        4. Inside of each job folder, many files are created,
            4.1. Creates a file named "options.json", a copy of the options.json file that was used to create the job, with the following modifications:
                - The axis1, axis2, and axis3 keys are modified to contain only the values of the axes that are used in the job.
                - The chunksize are put to -1 for all axes, meaning that the job will be run on the whole new range of the axis.
                - option['Header']['status'] = 'given'
                - option['Header']['job_number'] = i, where i is the job number.
            4.2. Creates a file run.sh from run_batch_template
        """
        # Step 1: Create main experiment folder
        Log.add_to_log(f"debug: {self.exp_name}")
        exp_folder_path = os.path.join(cst.path_to_Jobs, self.exp_name)
        os.makedirs(exp_folder_path, exist_ok=True)
        Log.add_to_log(f"(JM0002) Created/ensured experiment folder: {exp_folder_path}")
        
        # Step 2: Create axis-specific folder name based on ndim
        axis_folder_name_parts = []
        if self.ndim >= 1 and self.axis1.name is not None:
            axis_folder_name_parts.append(self.axis1.name)
        if self.ndim >= 2 and self.axis2.name is not None:
            axis_folder_name_parts.append(self.axis2.name)
        if self.ndim >= 3 and self.axis3.name is not None:
            axis_folder_name_parts.append(self.axis3.name)
        
        axis_folder_name = "_".join(axis_folder_name_parts)
        axis_folder_path = os.path.join(exp_folder_path, axis_folder_name)
        os.makedirs(axis_folder_path, exist_ok=True)
        Log.add_to_log(f"(JM0003) Created/ensured axis folder: {axis_folder_path}")
        
        # Step 3: Calculate total number of jobs and create job folders
        total_jobs = 1
        if self.ndim >= 1:
            total_jobs *= self.axis1.nchunks
        if self.ndim >= 2:
            total_jobs *= self.axis2.nchunks
        if self.ndim >= 3:
            total_jobs *= self.axis3.nchunks
        
        Log.add_to_log(f"(JM0004) Total jobs to create: {total_jobs}")
        
        # Generate all job combinations
        job_counter = 1  # Start job numbering from 1
        job_counter_init = 1
        
        # Get chunk ranges for each axis
        axis1_chunks = self._get_axis_chunks(self.axis1) if self.ndim >= 1 else [None]
        axis2_chunks = self._get_axis_chunks(self.axis2) if self.ndim >= 2 else [None]
        axis3_chunks = self._get_axis_chunks(self.axis3) if self.ndim >= 3 else [None]
        for i1, chunk1 in enumerate(axis1_chunks):
            for i2, chunk2 in enumerate(axis2_chunks):
                for i3, chunk3 in enumerate(axis3_chunks):
                    # Find next available job number
                    while os.path.exists(os.path.join(axis_folder_path, f"job{job_counter}")):
                        job_counter += 1
                        job_counter_init = job_counter
                    
                    job_folder_path = os.path.join(axis_folder_path, f"job{job_counter}")
                    os.makedirs(job_folder_path, exist_ok=True)
                    
                    # Step 4.1: Create modified options.json
                    modified_options = copy.deepcopy(self.option_json)
                    
                    # Modify axis parameters based on chunks
                    if self.ndim >= 1 and chunk1 is not None:
                        modified_options["parameters"]["Axis1"].update(chunk1)
                        modified_options["parameters"]["Axis1"]["chunksize"] = -1
                    
                    if self.ndim >= 2 and chunk2 is not None:
                        modified_options["parameters"]["Axis2"].update(chunk2)
                        modified_options["parameters"]["Axis2"]["chunksize"] = -1
                    
                    if self.ndim >= 3 and chunk3 is not None:
                        modified_options["parameters"]["Axis3"].update(chunk3)
                        modified_options["parameters"]["Axis3"]["chunksize"] = -1
                    
                    # Set job status and number
                    modified_options["Header"]["status"] = "given"
                    modified_options["Header"]["job_number"] = job_counter
                    
                    # Save modified options.json
                    options_json_path = os.path.join(job_folder_path, "options.json")
                    JFE.dump_json_data(modified_options, options_json_path)
                    
                    # Step 4.2: Create run.sh file
                    run_sh_content = self.run_batch_template(job_folder_path, job_number=job_counter)
                    run_sh_path = os.path.join(job_folder_path, "run.sh")
                    with open(run_sh_path, 'w') as f:
                        f.write(run_sh_content)
                    
                    # Make run.sh executable
                    os.chmod(run_sh_path, 0o755)
                    
                    Log.add_to_log(f"Created job{job_counter} in {job_folder_path}")
                    self.created_jobfolderpath_list.append(job_folder_path)
                    job_counter += 1
        
        Log.add_to_log(f"(JM0005) Successfully created {job_counter-job_counter_init} jobs")

    def _get_axis_chunks(self, axis):
        """
        Helper method to get chunk ranges for an axis
        """
        if axis.name is None or axis.nchunks <= 1:
            return [{"from": axis.from_, "to": axis.to}]
        
        chunks = []
        chunk_length = 2 * axis.pm * axis.chunksize
        """if abs((axis.to - axis.from_) % chunk_length) >= 10**-6:
            Log.add_to_log(f"Warning: Axis {axis.name} has a non-even range, chunks may not be evenly distributed.")
            # Adjust the last chunk to include any remainder
            axis.nchunks += 1"""
        for i in range(axis.nchunks):
            chunk_from = axis.from_ + i * chunk_length
            chunk_to = axis.from_ + (i + 1) * chunk_length
            if i == axis.nchunks - 1:  # Last chunk gets any remainder
                chunk_to = axis.to
            
            chunks.append({
                "from": chunk_from,
                "to": chunk_to if chunk_to <= axis.to else axis.to
            })
        
        return chunks

    def run_batch_template(self, job_folderpath, job_number=None):
        # creates the text to put for a batch calculation
        # Look jinja
        txt = (f"#!/bin/bash\n" +
                f"#SBATCH --job-name=j{job_number}{self.exp_name}\n" +
                f"#SBATCH --output={job_folderpath}/slurm-%j.out  # Path for the standard output file\n" +
                f"#SBATCH --partition=compute\n" +
                f"#SBATCH --time=8:00:00\n" +
                f"##SBATCH --ntasks=8\n" +
                f"#SBATCH --mem=0\n" +
                f"#SBATCH --account=mh0066\n" +
                f"\n" +
                f"set -eu\n" +
                f"source /sw/etc/profile.levante\n" +
                f"module purge\n" +
                f"module load python3/unstable\n\n"+
                f"\n" +
                f"job_folderpath=\"{job_folderpath}\"\n" +  # Declare the job folder path variable
                f"\n" +
                f"python3 {os.path.join(os.path.dirname(__file__), '..', 'main.py')} --job_folderpath=\"$job_folderpath\""
                )
        return txt

    def start_jobs(self):
        """
        Starts the jobs by running the batch script.
        This function assumes that the job folder has been created and contains a run.sh file.
        """
        if not self.are_jobs_created():     # just a safety check, should not be necessary
            Log.add_to_log("(JM0006) No job to start, either job number is not set or chunksize is not -1.")
            return
        for job_folderpath in self.created_jobfolderpath_list:
            run_sh_path = os.path.join(job_folderpath, "run.sh")

            if not os.path.exists(run_sh_path):
                Log.add_to_log(f"(JM0007) Run script {run_sh_path} does not exist, cannot start job.")
                return
            cmd = f"sbatch {run_sh_path}"
            jobi = job_folderpath.split('/')[-1]
            Log.add_to_log(f"(JM0008) Starting {jobi} with command: {cmd}")
            # subprocess.call(cmd, shell=True)

   