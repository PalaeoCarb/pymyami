import os
import re
import numpy as np
import pandas as pd
from glob import glob
from .helpers import MyAMI_parameter_file, expand_dims, match_dims, load_params, standard_seawater

def calc_seawater_ions(Sal=35., Na=None, K=None, Ca=None, Mg=None, Sr=None, Cl=None, BOH4=None, HCO3=None, CO3=None, SO4=None):
    """
    Returns modern seawater composition with given ions modified at specified salinity. 

    All units are mol/kg.

    NOTE: Assumes that the provided ionic concentrations are at Sal=35.

    Returns
    -------
    tuple of arrays
        Containing (cations, anions) in the order:
        cations = [H, Na, K, Mg, Ca, Sr]
        anions = [OH, Cl, B(OH)4, HCO3, HSO4, CO3, SO4] 

    """
    modified_cations = [None, Na, K, Mg, Ca, Sr]
    modified_anions = [None, Cl, BOH4, HCO3, None, CO3, SO4]

    m_cations, m_anions = standard_seawater()

    m_cations = np.full(
        (m_cations.size, *Sal.shape),
        expand_dims(m_cations, Sal)
        )

    m_anions = np.full(
        (m_anions.size, *Sal.shape),
        expand_dims(m_anions, Sal)
        )

    for i, m in enumerate(modified_cations):
        if m is not None:
            m_cations[i] = m
    
    for i, m in enumerate(modified_anions):
        if m is not None:
            m_anions[i] = m

    sal_factor = Sal / 35.

    return m_cations * sal_factor, m_anions * sal_factor

##########################################################################
# Functions tp calculate pitzer calculation matrices from parameter tables
##########################################################################

# dictionaries of ions containing their matrix indices
# positive ions H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
CA_IND = {
    'H': 0,
    'Na': 1,
    'K': 2,
    'Mg': 3,
    'Ca': 4,
    'Sr': 5
}

# negative ions  OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;
AN_IND = {
    'OH': 0,
    'Cl': 1,
    'B(OH)4': 2,
    'HCO3': 3,
    'HSO4': 4,
    'CO3': 5,
    'SO4': 6
}

# count ions
N_CA = len(CA_IND)  # H+=0; Na+=1; K+=2; Mg2+=3; Ca2+=4; Sr2+=5
N_AN = len(AN_IND)  # OH-=0; Cl-=1; B(OH)4-=2; HCO3-=3; HSO4-=4; CO3-=5; SO4-=6;

# build a regex for pulling ions out of salt names
recations = '|'.join(CA_IND.keys())
reanions = '|'.join(AN_IND.keys())
reanions = reanions.replace('(', '\(').replace(')', '\)')  # escape brackets
sm = re.compile('^(' + recations + ')[0-9]?\(?(' + reanions + ')\)?[0-9]?$')

def break_salt(s):
    """
    Returns positive and negative species names from salt or ion pair (X-Y).
    """
    if '-' in s:
        return s.split('-')
    
    match = sm.match(s)
    if match:
        gs = match.groups()
        return gs[0], gs[1]
    else:
        return None, None

# dictionary containing all valid ions
Iind = CA_IND.copy()
Iind.update(AN_IND)

# helper functions for converting tables into calculationg matrices

def filter_terms(tab, valid_ions):
    include = []
    for ions in tab.Pair.str.split('-'):
        include.append(~np.any([i not in valid_ions for i in ions]))

    return tab.loc[include]

def get_ion_index(ions):
    return tuple([Iind[k] for k in ions.split('-')])

# Load Parameter Tables for calculating pitzer parameters
TABLES = {}
fs = glob(MyAMI_parameter_file('Tab*.csv'))
for f in fs:
    fname = os.path.split(f)[-1].replace('.csv', '')
    TABLES[fname] = pd.read_csv(f, comment='#')
    TABLES[fname].fillna(0, inplace=True)

# remove unused pairs
TABA11 = filter_terms(TABLES['TabA11'], Iind)
TABA10 = filter_terms(TABLES['TabA10'], Iind)


# Table A1-9 equations for beta and C calculation
def EqA1(a, TK, TKinv, lnTK, **kwargs):
    # T in Kelvin
    return (
        a[0] +
        a[1] * TK +
        a[2] * TKinv +
        a[3] * lnTK +
        a[4] / (TK - 263) +
        a[5] * TK**2 +
        a[6] / (680 - TK) +
        a[7] / (TK - 227)
    )

def EqA2(a, TK, **kwargs):
    return a[0] + a[1] * TK + a[2] * TK**2

def EqA3A4(a, Tsub, **kwargs):
    return a[0] + a[1] * Tsub + a[2] * Tsub**2

def EqA5A6(a, TK, Tsub, **kwargs):
    return a[0] + a[1] * Tsub + a[2] * (TK - 303.15)**2

def EqA7(a, TK, **kwargs):
    PR, PJ, PLR = a[0:3]
    a = 88893.4225
    b = 8834524.63945833
    return (
        PR + PJ * (b - a * PLR) * (1 / TK - (1 / 298.15))
        + PJ / 6 * (TK**2 - a)
    )

# link tables to equations
EQ_TABLES = {
    'TabA1': EqA1,
    'TabA2': EqA2,
    'TabA3': EqA3A4,
    'TabA4': EqA3A4,
    'TabA5': EqA5A6,
    'TabA6': EqA5A6,
    'TabA7': EqA7,
    'TabA9': None,  # both special
    'TabSpecial': None  # special cases handled below
}

##########################################################################
# Special Case Equations noted in table subscripts and/or MyAMI_V1
##########################################################################
def EqA2_MgSO4(a, TK, **kwargs):
    return (
        a[0] * ((TK / 2) + (88804) / (2 * TK) - 298)
        + a[1] * ((TK**2 / 6) + (26463592) / (3 * TK) - (88804 / 2))
        + a[2] * (TK**3 / 12 + 88804 * 88804 / (4 * TK) - 26463592 / 3)
        + a[3]
        * ((TK**4 / 20) + 88804 * 26463592 / (5 * TK) - 88804 * 88804 / 4)
        + a[4] * (298 - (88804 / TK))
        + a[5]
    )

def EqA3A4_XHS(a, Tsub, TKinv, **kwargs):
    return a[0] + a[1] * TKinv + a[2] * Tsub**2

def EqA3_XSO3(a, TK, TKinv):
    return a[0] + a[1] * (TKinv - 1/298.15) + a[2] * np.log(TK / 298.15)

def EqA9_HCl(a, TK, **kwargs):
    return a[0] + a[1] * TK + a[2] / TK

def EqA9_HSO4(a, TK, **kwargs):
    TKsub3 = TK - 328.15
    return (
        a[0] + TKsub3 * 1e-3 * 
        (a[1] + TKsub3 * ((a[2] / 2) + TKsub3 * (a[3] / 6)))
        )

# dictionary of special cases
EQ_SPECIAL = {
    'TabA2': {
        'MgSO4': EqA2_MgSO4
    },
    'TabA3': {
        'KHS': EqA3A4_XHS,
        'Na2SO3': EqA3_XSO3,
        'NaHSO3': EqA3_XSO3,
    },
    'TabA4': {
        'KHS': EqA3A4_XHS,
    },
    'TabA9': {
        'H-Cl': EqA9_HCl,
        'H-SO4': EqA9_HSO4
    },
}

# Special case equations used in Mathis Hain's code.
def EQ_SPECIAL_Na2SO4_Moller(a, TK, lnTK, **kwargs):
    return (
        a[0] +
        a[1] * TK +
        a[2] / TK +
        a[3] * lnTK +
        a[4] / (TK - 263) +
        a[5] * TK**2 +
        a[6] / (680. - TK)
    )

def EQ_SPECIAL_MgHSO4(a, Tsub, **kwargs):
    return a[0] + a[1] * Tsub

# add these to dictionary
EQ_SPECIAL['TabSpecial'] = {
        'Na2SO4': EQ_SPECIAL_Na2SO4_Moller,
        'Mg(HSO4)2': EQ_SPECIAL_MgHSO4
    }

# Phi and Theta Equation
def EqA10(a, TK):
    """
    Calculate Phi and Theta parameters as a function of TK accoring to 
    """
    # a1 + a2 / T + a3 * T + a4 * (T - 298.15) + a5 * (T - 298.15)**2
    TKsub = TK - 298.15
    return a[0] + a[1] / TK + a[2] * 1e-4 * TK + a[3] * 1e-4 * TKsub + a[4] * 1e-6 * TKsub**2

# lambda and zeta function
def Eqn_A12(p, TK):
    a, b, c, d, e = p
    return a + b * TK + c * TK**2 + d / TK + e * np.log(TK)

##########################################################################
# Matrix creation functions
##########################################################################

def calc_beta_C(TK):
    """
    Calculate matrices of beta_0, beta_1, beta_2 and C_phi at given TK.

    Matrices are constructed from tables A1-A9 of Millero and Pierrot (1998; 
    doi:10.1023/A:1009656023546), with modifications implemented by Hain et 
    al (2015).

    Parameters
    ----------
    TK : array-like
        Temperature in Kelvin
    
    Returns
    -------
    dict of array-like
        Containing {beta_0, beta_1, beta_2, C_phi}
    """
    TKinv = 1. / TK
    lnTK = np.log(TK)
    Tsub = TK - 298.15

    # create blank parameter tables
    params = {k: np.zeros((N_CA, N_AN, *TK.shape)) for k in ['beta_0', 'beta_1', 'beta_2', 'C_phi']}

    # All except Table A8 - Temperature Sensitive
    for table in EQ_TABLES:
        # iterate through each parameter type and salt in each table.
        for (param, salt), g in TABLES[table].groupby(['Parameter', 'Salt']):
            p, n = break_salt(salt)  # identify which ions are involved
            if (p in CA_IND) and (n in AN_IND):
                # get the matrix indices of those ions
                pi = CA_IND[p]
                ni = AN_IND[n]
                
                eqn = EQ_TABLES[table]  # identify the correct equation
                if table in EQ_SPECIAL:  # does the table have special cases?
                    if salt in EQ_SPECIAL[table]:  # is this salt a special case?
                        eqn = EQ_SPECIAL[table][salt]  # if so, use the special equation.

                # calculate the parameter values and store them
                params[param][pi, ni] = eqn(a=g.values[0][2:], TK=TK, Tsub=Tsub, TKinv=TKinv, lnTK=lnTK)

    # Table A8 - Constants
    for i, row in TABLES['TabA8'].iterrows():
        p, n = break_salt(row.Salt)
        if (p in CA_IND) and (n in AN_IND):
            pi = CA_IND[p]
            ni = AN_IND[n]
            for param in params:
                params[param][pi, ni] = row[param]
    
    return params


def calc_Theta_Phi(TK):
    """
    Construct Theta and Phi matrices from Table A10 and A11 of Millero and Pierrot (1998).

    Parameters
    ----------
    TK : array-like
        Temperature in Kelvin
    
    Returns
    -------
    dict of array-like
        Containing {Theta_negative, Theta_positive, Phi_NNP, Phi_PPN}
    """

    # create empty arrays
    Theta_positive = np.zeros((N_CA, N_CA, *TK.shape))
    Theta_negative = np.zeros((N_AN, N_AN, *TK.shape))
    Phi_PPN = np.zeros((N_CA, N_CA, N_AN, *TK.shape))
    Phi_NNP = np.zeros((N_AN, N_AN, N_CA, *TK.shape))

    # Assign static values from Table A11
    for _, row in TABA11.iterrows():
        ions = row.Pair.split('-')
        index = get_ion_index(row.Pair)

        if ions[0] in CA_IND:
            if len(ions) == 2:
                Theta_positive[index] = row.Value
                Theta_positive[index[::-1]] = row.Value
            elif len(ions) == 3:
                Phi_PPN[index] = row.Value
                Phi_PPN[index[1], index[0], index[2]] = row.Value
        
        if ions[0] in AN_IND:
            if len(ions) == 2:
                Theta_negative[index] = row.Value
                Theta_negative[index[::-1]] = row.Value
            elif len(ions) == 3:
                Phi_NNP[index] = row.Value
                Phi_NNP[index[1], index[0], index[2]] = row.Value

    # Assign T-sensitive values from Table A10
    pnames = ['a1', 'a2', 'a3_e4', 'a4_e4', 'a5_e6']  # parameter names in TABle
    for _, row in TABA10.iterrows():
        ions = row.Pair.split('-')
        index = get_ion_index(row.Pair)
        
        a = row[pnames]  # identify parameters
        val = EqA10(a, TK)  # calculate value
        
        # assign value
        if ions[0] in CA_IND:
            if len(ions) == 2:
                Theta_positive[index] = val
                Theta_positive[index[::-1]] = val
            elif len(ions) == 3:
                Phi_PPN[index] = val
                Phi_PPN[index[1], index[0], index[2]] = val
        if ions[0] in AN_IND:
            if len(ions) == 2:
                Theta_negative[index] = val
                Theta_negative[index[::-1]] = val
            elif len(ions) == 3:
                Phi_NNP[index] = val
                Phi_NNP[index[1], index[0], index[2]] = val

    # Special cases that deviate from values in Millero and Pierrot (1998)
    special = {
        'Na-Ca-Cl': -7.6398 + -1.2990e-2 * TK + 1.1060e-5 * TK**2 + 1.8475 * np.log(TK),  # Spencer et al 1990
        'Mg-Ca-Cl': 4.15790220e1 + 1.30377312e-2 * TK - 9.81658526e2 / TK - 7.4061986 * np.log(TK),  # Spencer et al 1990
        'Cl-CO3': -0.092,  #Spencer et al 1990
        'CO3-OH': 0.1,  # http://www.aim.env.uea.ac.uk/aim/accent4/parameters.html
    }

    for ionstr, v in special.items():
        ions = ionstr.split('-')
        index = get_ion_index(ionstr)
        if ions[0] in CA_IND:
            if len(ions) == 2:
                Theta_positive[index] = v
                Theta_positive[index[::-1]] = v
            elif len(ions) == 3:
                Phi_PPN[index] = v
                Phi_PPN[index[1], index[0], index[2]] = v
        if ions[0] in AN_IND:
            if len(ions) == 2:
                Theta_negative[index] = v
                Theta_negative[index[::-1]] = v
            elif len(ions) == 3:
                Phi_NNP[index] = v
                Phi_NNP[index[1], index[0], index[2]] = v

    return {
        'Theta_negative': Theta_negative, 
        'Theta_positive': Theta_positive, 
        'Phi_NNP': Phi_NNP, 
        'Phi_PPN': Phi_PPN
        }

def calc_lambda_zeta(TK):
    """
    Return lambda and zeta matrices from Table A12 of Millero and Pierrot (1998).

    Parameters
    ----------
    TK : arra-like
        Temperature in Kelvin

    Returns
    -------
    dict
        Containing {lambdaCO2, zetaCO2, lambdaB, zetaB}
    """
    cations = ['H', 'Na', 'K', 'Mg', 'Ca']
    anions = ['Cl', 'SO4']
    ions = cations + anions

    TabA12 = TABLES['TabA12']
    
    lambdaCO2 = np.zeros((7, *TK.shape))
    for i, ion in enumerate(ions):
        p = TabA12.loc[(TabA12.Parameter == 'lambda_CO2') & (TabA12.i == ion), ['a', 'b', 'c', 'd', 'e']]
        if p.size > 0:
            lambdaCO2[i] = Eqn_A12(p.values[0], TK)
            
    zetaCO2 = np.zeros([2, 5, *TK.shape])

    for i, cation in enumerate(cations):
        for j, anion in enumerate(anions):
            p = TabA12.loc[(TabA12.Parameter == 'zeta_CO2') & (TabA12.i == cation) & (TabA12.j == anion), ['a', 'b', 'c', 'd', 'e']]
            if p.size > 0:
                zetaCO2[j, i] = Eqn_A12(p.values[0], TK)

    lambdaB = np.zeros((7))
    for i, ion in enumerate(ions):
        p = TabA12.loc[(TabA12.Parameter == 'lambda_BOH3') & (TabA12.i == ion), ['a']]
        if p.size > 0:
            lambdaB[i] = p.values
            
    zetaB = np.zeros([2, 5])
    for i, cation in enumerate(cations):
        for j, anion in enumerate(anions):
            p = TabA12.loc[(TabA12.Parameter == 'zeta_BOH3') & (TabA12.i == cation) & (TabA12.j == anion), ['a']]
            if p.size > 0:
                zetaB[j, i] = p.values
    
    return {
        'lambdaCO2': lambdaCO2,
        'zetaCO2': zetaCO2,
        'lambdaB': lambdaB,
        'zetaB': zetaB
    }

def PitzerParams(TK):
    """
    Return Pitzer params for given T (Kelvin).
    
    Parameters
    ----------
    TK : array-like
        Temperature in Kelvin
        
    Returns
    -------
    dict of arrays
        with keys: beta_0, beta_1, beta_2, C_phi, Theta_negative, Theta_positive, Phi_NNP, Phi_PPN
    """
    if isinstance(TK, (float, int)):
        TK = np.asanyarray(TK)
    
    out = {}
    out.update(calc_beta_C(TK))
    out.update(calc_Theta_Phi(TK))
    # out.update(calc_lambda_zeta(TK))

    return out
