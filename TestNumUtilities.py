import unittest

import numpy as np

from NumUtilities import return_over_number_periods


class MyTestCase(unittest.TestCase):
    def test_return_over_number_periods(self):
        n = 3
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.21, 1.21, 1.21])
        xout, yout = return_over_number_periods(n, x, y)
        assert np.array_equal(xout, np.array([0, 1, 2, 3, 4, 5]))
        assert np.array_equal(yout.round(3), np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).round(3))


if __name__ == '__main__':
    unittest.main()
