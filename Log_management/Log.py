import os

import Constants as cst
import time


# DEBUG
GLOBAL_DEBUG_TEXT = ''

LEVEL = 0
# Level definition
DEBUG = 0
NORMAL = 2
WARNING = 3
HIGHEST = 5
LOG_FILE_NAME = 'Log.txt'

'''
def init_log(): Use this function to initiate the log file in the main path. This log file will be used for Debug, just
in case.

'''


def init_log(job_folderpath, filename):
    # Modifying the name of the folder containing
    f = open(os.path.join(job_folderpath, filename), "w+")
    f.close()
    print(f'The job folder path : {os.path.join(job_folderpath, filename)}')
    cst.modify_job_folderpath(os.path.join(job_folderpath, filename))


'''
def add_to_log(txt): This function adds data to the log file and makes a carriage return
in:
		-txt : the string to add for debug
		-level : The level of importance of the warning
'''


def add_to_log(txt, start="date", end="\n", level=DEBUG):
    """
    def add_to_log(txt): This function adds data to the log file and makes a carriage return
    in:
            -txt : the string to add for debug
            -level : The level of importance of the warning
    """
    if level >= LEVEL:
        # save the last directory
        root = os.getcwd()
        # dir_path = os.path.dirname(os.path.realpath(__file__))

        # create the start str
        if start == "date":
            now = time.strftime("%Y %b %d %H:%M:%S")
            now = str("%s" % now)
            start = now + ' > '
        ################### SEND LOG NAME
        text_to_write = start + txt + end
        with open(cst.get_job_folderpath(), "a") as file1:  # add your lines
            file1.write(text_to_write)
        print(start + txt, end=end)

        # get back to the last directory
        os.chdir(root)
