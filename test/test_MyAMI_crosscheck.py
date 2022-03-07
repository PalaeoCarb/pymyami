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

        print('Comparing Ks to MyAMI_V1')
        for k in Ks:
            v1 = check[k]
            new = new_Fcorr[k]

            maxpercentdiff = 100 * np.max(np.abs((v1 - new) / v1))
            
            print(f'  {k}: {maxpercentdiff:.2f}%')
            self.assertLess(maxpercentdiff, 0.4, msg=f'Maximum difference in {k} correction factor too large: {maxpercentdiff}%')

# class MyAMI_V2_crosscheck(unittest.TestCase):
#     """Compare Fcorr factors"""

#     def test_Fcorr(self):
#         # generate check grid
#         n = 5
#         T, S, Mg, Ca = np.meshgrid(
#             np.linspace(0, 40, n),
#             np.linspace(30, 40, n),
#             np.linspace(0,0.06,n),
#             np.linspace(0,0.06,n)
#         )

#         old_Fcorr = MyAMI_Fcorr(XmCa=Ca, XmMg=Mg, TempC=T, Sal=S)
#         new_Fcorr = MyAMI.calc_Fcorr(Sal=S, TempC=T, Mg=Mg, Ca=Ca)

#         for k in old_Fcorr:
#             old = old_Fcorr[k]
#             new = new_Fcorr[k]

#             maxdiff = np.max(np.abs(old - new))

#             self.assertAlmostEqual(0, maxdiff, places=12, msg=f'Maximum difference in {k} correction factor too large: {maxdiff}')

#     def test_approx_Fcorr(self):
#         n = 5
#         TempC, Sal, Mg, Ca = np.meshgrid(
#             np.linspace(0, 40, n),
#             np.linspace(30, 40, n),
#             np.linspace(0,0.06,n),
#             np.linspace(0,0.06,n)
#         )   

#         Fcorr_calc = MyAMI.calc_Fcorr(TempC=TempC, Sal=Sal, Mg=Mg, Ca=Ca)
#         Fcorr_approx = MyAMI.approximate_Fcorr(TempC=TempC, Sal=Sal, Mg=Mg, Ca=Ca)
        
#         for k in Fcorr_calc:
#             maxpcdiff = np.max(np.abs(100 * (Fcorr_approx[k] - Fcorr_calc[k]) / Fcorr_calc[k]))
#             self.assertLess(maxpcdiff, 0.6, msg=f'Approximate {k} is greater than 0.2% from calculated {k}.')


if __name__ == "__main__":
    unittest.main()
