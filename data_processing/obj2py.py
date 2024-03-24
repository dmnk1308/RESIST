from vedo import *
import scipy.io
import numpy as np

def read_mat(path, targets=True):
    ''' 
    Reads the mat file at path and returns the array data.
    '''
    mat = scipy.io.loadmat(path)
    key = list(mat.keys())[3]
    data = mat[key]
    if isinstance(data, np.ndarray):
        data[np.isnan(data)] = 0
        data = data.T
        return data
    else:
        print('Check .mat file!')

def read_egt(fullname_egt=None):
    '''
    Reads the EGT data from the file fullname_egt and returns the data as a matrix.
    '''
    fid = open(fullname_egt, "r")
    if fid == -1:
        print(f"{fullname_egt} does not exist!")
        raise ValueError(f"Could not open the file {fullname_egt}!")
    egtdata = np.fromfile(fid, dtype = np.float32)
    egtdata = egtdata.reshape(int(np.sqrt(egtdata.shape[0])),-1)
    return egtdata

def read_get(fullname_get=None):
    ''' 
    Reads the GET data from the file fullname_get and returns the data as a matrix.
    '''
    fid = open(fullname_get, "r")
    if fid == -1:
        print(f"{fullname_get} does not exist!")
        raise ValueError(f"Could not open the file {fullname_get}!")
    getdata = np.fromfile(fid, dtype = np.float32)
    getdata = getdata.reshape(256,-1)
    getdata = getdata[~np.isnan(getdata)]
    return getdata


