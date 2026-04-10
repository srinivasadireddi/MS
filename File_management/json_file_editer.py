import json
from Log_management import Log as Log
import os


'''
def write_in_json(wtwrite, j1='', j2='', j3='', file="params.json"): this function write in json files, for the purpose
of the program, the usual file is a params.json one.
'''


def write_in_json(wtwrite, j=[], file="params.json"):
    retour = 0
    with open(file, "r") as read_file:
        data = json.load(read_file)
    if len(j) == 0:
        data = wtwrite
    elif len(j) == 1:
        data[j[0]] = wtwrite
    elif len(j) == 2:
        # In params.json, the 'n' param might not exist
        if j[1] == 'n' and wtwrite == '':
            data[j[0]].pop([j[1]], None)
        else:
            data[j[0]][j[1]] = wtwrite
    elif len(j) == 3:
        data[j[0]][j[1]][j[2]] = wtwrite
    else:
        print('An error occurred in the write function in .json files (JFE0001)')
    with open(file, "w") as write_file:
        json.dump(data, write_file, indent=4)
        # adding indent=4 will make the json files hard to write on SC




def read_in_json(j=[], file="params.json"):
    retour = 0
    with open(file, "r") as read_file:
        data = json.load(read_file)
    retour = read_data(data, j)
    return retour


def read_data(data, j=[]):
    print(j)
    if len(j) == 0:
        retour = data
    elif len(j) == 1:
        retour = data[j[0]]
    elif len(j) == 2:
        if j[1] == 'n' and 'n' not in data[j[0]]:
            retour = ''
        else:
            retour = data[j[0]][j[1]]
    elif len(j) == 3:
        retour = data[j[0]][j[1]][j[2]]
    elif len(j) == 4:
        retour = data[j[0]][j[1]][j[2]][j[3]]
    else:
        print('An error occurred in the read function in .json files (JFE0002)')
    return retour


def get_json_data(file="params.json"):
    with open(file, "r") as read_file:
        data = json.load(read_file)
    return data


def dump_json_data(data, file="params.json"):
    with open(file, "w") as write_file:
        json.dump(data, write_file, indent=4)     # overwrites the file


def modify_json_data(data, wtwrite, j=[]):
    print(wtwrite)
    try:
        val = eval(wtwrite)
    except:
        val = wtwrite
    if len(j) == 0:
        data = val
    elif len(j) == 1:
        data[j[0]] = val
    elif len(j) == 2:
        # In params.json, the 'n' param might not exist
        if j[1] == 'n' and val == '':
            data[j[0]].pop(j[1], None)
        else:
            data[j[0]][j[1]] = val
    elif len(j) == 3:
        data[j[0]][j[1]][j[2]] = val
    elif len(j) == 4:
        data[j[0]][j[1]][j[2]][j[3]] = val
    else:
        print('An error occurred in the write function in .json files (JFE0003)')


def make_json_tab_from_txt(file):
    with open(file, "r") as my_file:
        str = my_file.read()
    list = str.split('\n')
    tab = []
    for x in list[:len(list)-1]:
        tab.append(x.split(' '))
    return tab


'''
def compare_json(json1, json2): This function compares 2 json data list made by the same function of this very same
file. 
'''


def compare_json(json1, json2, compare, verify=False):
        status = 'same'
        for x in compare:
            A, B = read_data(json1, x), read_data(json2, x)
            
            try:
                
                Aa = str("%.2f" % A)
                Bb = str("%.2f" % B)
                if Aa != Bb:
                    status = 'different'
                if verify:
                    print('compare ' + str(A) + ' with ' + str(B) + ' status is ' + status + '  (JFE0006)')
            except:
                if A != B:
                    status = 'different'
        if verify:
            print('The status is : ' , status)
        return status

