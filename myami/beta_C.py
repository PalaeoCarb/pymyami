from curses.ascii import TAB
from distutils.command.build import build
from xml.etree.ElementTree import PI
import numpy as np
from .params import TABLES, Pind, Nind, Iind, N_anions, N_cations  # constants
from .params import build_salt, break_salt

# def construct_beta_0():

TK = 25 + 298.15
TKinv = 1. / TK
lnTK = np.log(TK)
# ln_of_Tdiv29815 = np.log(T / 298.15)
Tpower2 = TK**2.
Tpower3 = TK**3.
Tpower4 = TK**4.
TC = TK - 298.15

def EqA1(a, T, Tinv, lnT):
    # T in Kelvin
    return (
        a[0] +
        a[1] * T +
        a[2] * Tinv +
        a[3] * lnT +
        a[4] / (T - 263) +
        a[5] * T**2 +
        a[6] / (680 - T) +
        a[7] / (T - 227)
    )

TABA1 = TABLES['TabA1']

params = {k: np.zeros((N_cations, N_anions)) for k in ['beta_0', 'beta_1', 'beta_2', 'C_phi']}

for (param, salt), g in TABA1.groupby(['Parameter', 'Salt']):
    p, n = break_salt(salt)
    pi = Pind[p]
    ni = Nind[n]
    
    print(p, n, g.values[0][2:])
    params[param][pi, ni] = EqA1(g.values[0][2:], TK, TKinv, lnTK)
                    

# if __name__ == "__main__":
#     print(TABLES['TabA1'])