import unittest

import numpy
import numpy as np
from numpy.ma.testutils import approx
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from DataPlotUtil import scatter_plot_with_regression_line
from DfUtil import scale_to_start_value
from NumUtilities import return_over_number_periods, yield_return, moving_average, first_non_nan, replace_leading_nan

class MyTestCase(unittest.TestCase):
    def test_scatter_plot_with_regression(self):
        x = np.arange(0,6,dtype=float)
        y = np.array([3.0,3.9,5.3,6.9,7.3,8.1])
        scatter_plot_with_regression_line(x, y)
        scatter_plot_with_regression_line(x, y, 2)

