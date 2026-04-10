import os


path_to_compileddata = os.path.join(os.path.dirname(__file__), '../', 'CompiledData')
path_to_main = os.path.join(os.path.dirname(__file__))
path_to_Jobs = os.path.join(path_to_main, 'Jobs')

options_json_filename = 'options.json'
path_to_r02b09_gridfile = '/pool/data/ICON/grids/public/mpim/0055/icon_grid_0055_R02B09_G.nc'

qmin = 10**(-12)


def get_job_folderpath():
    return job_folderpath


def modify_job_folderpath(Name):
    global job_folderpath
    if Name == '':
        job_folderpath = os.path.join(os.path.dirname(__file__), 'Log_management')
    else:
        job_folderpath = os.path.join(Name)


def get_job_number():
    return job_number


def modify_job_number(i):
    global job_number
    job_number = i



# Changing the options
def modify_op_status(op,val=''):
    op["Header"]["status"] = val