import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from NumUtilities import return_over_number_periods, yield_return


class MyTestCase(unittest.TestCase):
    def test_return_over_number_periods(self):
        n = 3
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.21, 1.21, 1.21])
        xout, yout = return_over_number_periods(n, x, y)
        assert np.array_equal(xout, np.array([0, 1, 2, 3, 4, 5]))
        assert_almost_equal(yout, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    def test_yield_return(self):
        x = np.array([0.1,0.1,0.1])
        initial = 1.0
        yout = yield_return(initial,x)
        assert_almost_equal(yout,[1.0,1.1,1.21])

if __name__ == '__main__':
    unittest.main()
