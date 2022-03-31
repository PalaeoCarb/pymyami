import os
import unittest
import numpy as np
import pandas as pd

import pymyami

class MyAMI_V1_crosscheck(unittest.TestCase):

    def test_Fcorr(self):
        checkfile = os.path.join(os.path.dirname(__file__), 'data/MyAMI_V1_Fcorr_checkvals.csv')
        print(checkfile)
        check = pd.read_csv(checkfile, index_col=0)
        check.columns = ['T', 'S', 'Ca', 'Mg', 'KspC', 'K1', 'K2', 'KW', 'KB', 'KspA', 'K0', 'KS']

        new_Fcorr = pymyami.calc_Fcorr(Sal=check.S.values, TempC=check['T'].values, Ca=check.Ca.values, Mg=check.Mg.values)

        Ks = 'K0', 'K1', 'K2', 'KW', 'KB', 'KspA', 'KspC', 'KS'

        print('Comparing Fcorr to MyAMI_V1 (must be <0.4% max difference)')
        for k in Ks:
            v1 = check[k]
            new = new_Fcorr[k]

            rdiff = (v1 - new) / v1  # relative difference
            
            maxpercentdiff = 100 * np.max(np.abs(rdiff))
            avgpercentdiff = 100 * np.mean(rdiff)
            
            print(f'  {k}: {maxpercentdiff:.2f}% max, {avgpercentdiff:.2f}% avg')
            self.assertLess(maxpercentdiff, 0.4, msg=f'Maximum difference in {k} correction factor too large: {maxpercentdiff}%')

if __name__ == "__main__":
    unittest.main()
