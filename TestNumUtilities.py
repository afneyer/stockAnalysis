import unittest

import numpy
import numpy as np
from numpy.ma.testutils import approx
from numpy.testing import assert_almost_equal

from NumUtilities import return_over_number_periods, yield_return, moving_average, first_non_nan, replace_leading_nan


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

    def test_moving_average(self):
        x = [1.0,1.1,1.2,1.3,1.4,1.5]
        y = moving_average(x,3)
        print(y)
        assert approx(y,[1.1,1.2,1.3,1.4]).all()

        x.append(np.nan)
        print(x)
        y = moving_average(x,3)
        assert approx(y[:4],[1.1,1.2,1.3,1.4]).all()
        assert np.isnan(y[4])

    def test_first_non_nan(self):
        x = [numpy.NaN, numpy.NaN, numpy.NaN, 1.0, 1.2, 1.3, 1.4, 1.5, numpy.NaN, numpy.NaN]
        i = first_non_nan(x)
        assert i == 3

    def test_replace_leading_nan(self):
        x = [numpy.NaN, numpy.NaN, numpy.NaN, 1.0, 1.2, 1.3, 1.4, 1.5, numpy.NaN, numpy.NaN]
        y = replace_leading_nan(x,0.0)
        assert_almost_equal(y, [0.,  0.,  1.,  1.2, 1.3, 1.4, 1.5, numpy.NaN, numpy.NaN])
        print(y)



if __name__ == '__main__':
    unittest.main()
