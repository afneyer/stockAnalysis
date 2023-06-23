import unittest

import numpy
import numpy as np
from numpy.ma.testutils import approx
from numpy.testing import assert_almost_equal

from NumUtilities import return_over_number_periods, yield_return, moving_average, first_non_nan, replace_leading_nan
from StockTotalReturnGraph import totalReturnGraph


class MyTestCase(unittest.TestCase):
    def test_vnq_rez_spy(self):
        totalReturnGraph(['spy','rez','vnq'],period='10y')

