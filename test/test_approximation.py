import unittest
import numpy as np
from pymyami import approximate_Fcorr, calc_Fcorr

import matplotlib.pyplot as plt

# PARAMETERS
TOLERANCE = 0.2  # percent
N = 10000

class TestApproximation(unittest.TestCase):
    
    def test_approximate_Fcorr(self):
        # set random state
        np.random.seed(42)

        # generate test conditions
        TempC = np.random.uniform(low=0, high=40, size=N)
        Sal = np.random.uniform(low=30, high=40, size=N)
        Mg = np.random.uniform(low=0, high=0.06, size=N)
        Ca = np.random.uniform(low=0, high=0.06, size=N)

        print(f'\n\nChecking approximation function (max relative difference <{TOLERANCE}%)...\n')
        # calculate Fcorr using each method
        direct = calc_Fcorr(TempC=TempC, Sal=Sal, Mg=Mg, Ca=Ca)
        approx = approximate_Fcorr(TempC=TempC, Sal=Sal, Mg=Mg, Ca=Ca)

        # test Ks
        for k in direct:
            diff = approx[k] - direct[k]
            rdiff = diff / direct[k]
            
            maxrdiff = 100 * np.max(np.abs(rdiff))
            avgrdiff = 100 * np.mean(rdiff)
            
            print(f'  {k}: {maxrdiff:.2f}% max, {avgrdiff:.2f}% avg')
            
            with self.subTest(k=k):
                self.assertLess(maxrdiff, TOLERANCE, msg=f'Maximum difference in {k} correction factor too large: {maxrdiff}%')
        