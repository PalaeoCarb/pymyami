"""
Helper functions used throughout MyAMI.
"""
import json
import numpy as np
import pandas as pd
import pkg_resources as pkgrs

def MyAMI_parameter_file(fname=''):
    return pkgrs.resource_filename('pymyami', f'parameters/{fname}')

def expand_dims(orig, target):
    """
    Adds additional dimensions to orig so it can be broadcast on target.
    """
    return np.expand_dims(orig, tuple(range(orig.ndim, target.ndim + orig.ndim)))

def match_dims(orig, target):
    """
    Adds additional dimensions to orig to match the number of dimensions in target.
    """
    return np.expand_dims(orig, tuple(range(orig.ndim, target.ndim)))

def shape_matcher(*args):
    """
    Returns all given arrays as the same shape.
    """
    biggest = 0
    shape = None
    for a in args:
        if a is not None:
            a = np.asanyarray(a)
            if a.size > biggest:
                biggest = a.size
                shape = a.shape
    
    for a in args:
        if a is not None:
            a = np.asanyarray(a)
            if a.shape != shape:
                yield np.full(shape, a)
            else:
                yield a
        else:
            yield a

def load_params(param_file, asarrays=True):
    """
    Load parameters from a json file and convert the entries to np.ndarray.

    Parameters
    ----------
    param_file : string
        The name of the file within MyAMI/resources to load.
    asarrays : bool, optional
        If true, entries are converted to np.ndarray, by default True

    Returns
    -------
    dict
        The contents of the json file as a dict.
    """
    with open(MyAMI_parameter_file(param_file), 'r') as f:
        params = json.load(f)
    
    if asarrays:
        return {k: np.array(v) for k, v in params.items()}
    return params

def calc_Istr(Sal):
    """
    Calculation ionic strength from Salinity
    """
    return 19.924 * Sal / (1000 - 1.005 * Sal)

def standard_seawater(S=35.):
    """
    Return modern seawater ionic composition at specified salinity
    in units of mol/kg.

    Parameters
    ----------
    S : array-like
        Salinity in PSU

    Returns
    -------
    tuple of arrays
        Containing (cations, anions) in the order:
        cations = [H, Na, K, Mg, Ca, Sr]
        anions = [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4] 
    """

    sw = pd.read_csv(MyAMI_parameter_file('standard_seawater.csv'), index_col=0, comment='#')
    
    cation_concs = sw.loc[['H', 'Na', 'K', 'Mg', 'Ca', 'Sr'], :].values.T * S / 35
    anion_concs = sw.loc[['OH', 'Cl', 'BOH4', 'HCO3', 'HSO4', 'CO3', 'SO4'], :].values.T * S / 35

    return cation_concs, anion_concs

def calc_KS(TK=25., Sal=35., Istr=None):
    lnTK = np.log(TK)
    p = [141.328, -4276.1, -23.093, -13856, 324.57, -47.986, 35474, -771.54, 114.723, -2698, 1776]
    
    return np.exp(
        p[0]
        + p[1] / TK
        + p[2] * lnTK
        + np.sqrt(Istr) * (p[3] / TK + p[4] + p[5] * lnTK)
        + Istr * (p[6] / TK + p[7] + p[8] * lnTK)
        + p[9] / TK * Istr * np.sqrt(Istr)
        + p[10] / TK * Istr ** 2
        + np.log(1 - 0.001005 * Sal)
    )
    
def calc_KF(TK=25., Sal=35.):
    p = [874, -9.68, 0.111]  # from Best Practices guide.
    # p = [1590.2, 12.641, 1.525]  # used in MyAMI_V1
    
    return np.exp(
        p[0] / TK + 
        p[1] + 
        p[2] * Sal**0.5
    )