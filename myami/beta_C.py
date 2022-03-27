from curses.ascii import TAB
from distutils.command.build import build
from xml.etree.ElementTree import PI
from click import Tuple
import numpy as np
from .params import TABLES, Pind, Nind, Iind, N_anions, N_cations, get_ion_index  # constants
from .params import build_salt, break_salt
from .params import PitzerParams

# TK = np.array([15, 25, 35]) + 298.15
TK = np.array([25]) + 298.15
TKinv = 1. / TK
lnTK = np.log(TK)
TKpower2 = TK**2.
TKpower3 = TK**3.
TKpower4 = TK**4.
Tsub = TK - 298.15

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
    return (
        PR + PJ * (8834524.63945833 - 88893.4225 * PLR) * (1 / TK - (1 / 298.15))
        + PJ / 6 * (TK**2 - 88893.4225)
    )


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

def EqA9_HCl(a, TK, **kwargs):
    return a[0] + a[1] * TK + a[2] / TK

def EqA9_HSO4(a, TK, **kwargs):
    TKsub3 = TK - 328.15
    return (
        a[0] + TKsub3 * 1e-3 * 
        (a[1] + TKsub3 * ((a[2] / 2) + TKsub3 * (a[3] / 6)))
        )

def EqSpecial_Na2SO4_Moller(a, TK, lnTK, **kwargs):
    return (
        a[0] +
        a[1] * TK +
        a[2] / TK +
        a[3] * lnTK +
        a[4] / (TK - 263) +
        a[5] * TK**2 +
        a[6] / (680. - TK)
    )

def EqSpecial_MgHSO4(a, Tsub, **kwargs):
    return a[0] + a[1] * Tsub


TabEqs = {
    'TabA1': EqA1,
    'TabA2': EqA2,
    'TabA3': EqA3A4,
    'TabA4': EqA3A4,
    'TabA5': EqA5A6,
    'TabA6': EqA5A6,
    'TabA7': EqA7,
    'TabA9': None,  # both special
    'TabSpecial': None
}

# Special Case Equations noted in table subscripts

def EqA3A4_XHS(a, Tsub, TKinv, **kwargs):
    return a[0] + a[1] * TKinv + a[2] * Tsub**2

def EqA3_XSO3(a, TK, TKinv):
    return a[0] + a[1] * (TKinv - 1/298.15) + a[2] * np.log(TK / 298.15)

EqSpecial = {
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
    'TabSpecial':{
        'Na2SO4': EqSpecial_Na2SO4_Moller,
        'Mg(HSO4)2': EqSpecial_MgHSO4
    }
}

params = {k: np.zeros((N_cations, N_anions, *TK.shape)) for k in ['beta_0', 'beta_1', 'beta_2', 'C_phi']}

for table in TabEqs:
    # print(table)
    for (param, salt), g in TABLES[table].groupby(['Parameter', 'Salt']):
        # print(salt)
        p, n = break_salt(salt)
        if (p in Pind) and (n in Nind):
            pi = Pind[p]
            ni = Nind[n]
            
            eqn = TabEqs[table]
            if table in EqSpecial:
                if salt in EqSpecial[table]:
                    eqn = EqSpecial[table][salt]             

            if (p == 'Mg') & (n == 'HSO4'):
                print(table, param, g.values[0][2:], eqn(a=g.values[0][2:], TK=TK, Tsub=Tsub, TKinv=TKinv, lnTK=lnTK))
            # if (table == 'TabA1') & (salt == 'NaCl'):
            #     print(param, p, n, g.values[0][2:])
            #     print(eqn(a=g.values[0][2:], TK=TK, Tsub=Tsub, TKinv=TKinv, lnT=lnTK))
            params[param][pi, ni] = eqn(a=g.values[0][2:], TK=TK, Tsub=Tsub, TKinv=TKinv, lnTK=lnTK)

# Table A8 - constants
for i, row in TABLES['TabA8'].iterrows():
    p, n = break_salt(row.Salt)
    if (p in Pind) and (n in Nind):
        pi = Pind[p]
        ni = Nind[n]
        for param in params:
            params[param][pi, ni] = row[param]



test_params = PitzerParams(T=TK)

Prind = {v:k for k, v in Pind.items()}
Nrind = {v:k for k, v in Nind.items()}

def compare_table(new, ref):
    ind = np.abs(new - ref) > 1e-12
    locs = np.argwhere(ind)
    
    for loc in locs:
        loc = tuple(loc)
        iP = Prind[loc[0]]
        iN = Nrind[loc[1]]
        iTK = TK[loc[2]]

        print(f'  {iP}-{iN} at {iTK}: {ref[loc]} vs {new[loc]}')

    # print(ref[Tuple(loc)])

for p in ['beta_0', 'beta_1', 'beta_2', 'C_phi']:
    diff = np.abs(params[p] - test_params[p])
    if np.all(diff < 1e-10):
        print(f'{p}:   OK')
    else:
        print(p)
        compare_table(params[p], test_params[p])


# if __name__ == "__main__":
    # print('yup')