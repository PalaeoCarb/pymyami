import unittest
import numpy as np
from pymyami import approximate_Fcorr, calc_Fcorr

class TestInputs(unittest.TestCase):
    
    def test_NDarray_input(self):
        # set random state
        np.random.seed(42)

        # generate test conditions
        TempC = np.random.uniform(low=30, high=40, size=(3,4))

        direct = calc_Fcorr(TempC=TempC)
    