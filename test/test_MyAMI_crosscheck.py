import os
import unittest
import numpy as np
import pandas as pd

import myami

class MyAMI_V1_crosscheck(unittest.TestCase):

    def test_Fcorr(self):
        checkfile = os.path.join(os.path.dirname(__file__), 'data/MyAMI_V1_Fcorr_checkvals.csv')
        print(checkfile)
        check = pd.read_csv(checkfile, index_col=0)
        check.columns = ['T', 'S', 'Ca', 'Mg', 'KspC', 'K1', 'K2', 'KW', 'KB', 'KspA', 'K0', 'KS']

        new_Fcorr = myami.calc_Fcorr(Sal=check.S.values, TempC=check['T'].values, Ca=check.Ca.values, Mg=check.Mg.values)

        Ks = 'K0', 'K1', 'K2', 'KW', 'KB', 'KspA', 'KspC', 'KS'

        print('Comparing Ks to MyAMI_V1 (must be <0.4% different)')
        for k in Ks:
            v1 = check[k]
            new = new_Fcorr[k]

            maxpercentdiff = 100 * np.max(np.abs((v1 - new) / v1))
            
            print(f'  {k}: {maxpercentdiff:.2f}%')
            self.assertLess(maxpercentdiff, 0.4, msg=f'Maximum difference in {k} correction factor too large: {maxpercentdiff}%')

if __name__ == "__main__":
    unittest.main()
