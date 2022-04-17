""""
User-facing function for calculating K correction factors using MyAMI
"""

from .helpers import shape_matcher
from .pitzer import calc_gKs
from .params import PitzerParams


def calc_Fcorr(Sal=35., TempC=25., Na=None, K=None, Mg=None, Ca=None, Sr=None, Cl=None, BOH4=None, HCO3=None, CO3=None, SO4=None):
    """
    Calculate K correction factors as a fn of temp and salinity that can be applied to empirical Ks

    Parameters
    ----------
    TempC : array-like
        Temperature in Celcius
    Sal : array-like
        Salinity in PSU
    Na, K, Mg, Ca, Sr : array-like
        Cation concentrations in mol/kg. If None, mean ocean water values are used.
    Cl, BOH4, HCO3, CO3, SO4 : array-like
        Anion concentrations in mol/kg. If None, mean ocean water values are used.


    Returns
    -------
    dict 
        Correction factors (Fcorr) to be applied to empirical K values,
        where K_corr = K_cond * F_corr.
    """

    # ensure all inputs are the same shape
    TempC, Sal, Na, K, Mg, Ca, Sr, Cl, BOH4, HCO3, CO3, SO4 = shape_matcher(TempC, Sal, Na, K, Mg, Ca, Sr, Cl, BOH4, HCO3, CO3, SO4)

    # Calculate pitzer parameters at given temperature
    # {beta_1, beta_2, beta_3, C_phi, Theta_positive, Theta_negative, Phi_NNP, Phi_PPN}
    pitzer_params = PitzerParams(TempC + 273.15)

    # Calculate gK's for modern (mod) and experimental (x) seawater composition
    mod = calc_gKs(TempC, Sal, **pitzer_params)
    
    X = calc_gKs(TempC, Sal, 
                      Na=Na, K=K, Mg=Mg, Ca=Ca, Sr=Sr, Cl=Cl, 
                      BOH4=BOH4, HCO3=HCO3, CO3=CO3, SO4=SO4, 
                      **pitzer_params)

    # Calculate conditional K's predicted for seawater composition X
    F_dict = {k.replace('g', ''): X[k] / mod[k] for k in mod}

    return F_dict

