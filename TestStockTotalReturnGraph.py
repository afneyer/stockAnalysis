import unittest

import numpy
import numpy as np
from numpy.ma.testutils import approx
from numpy.testing import assert_almost_equal

from NumUtilities import return_over_number_periods, yield_return, moving_average, first_non_nan, replace_leading_nan
from StockTotalReturnGraph import totalReturnGraph


class TestTotalReturnGraph(unittest.TestCase):
    def test_vnq_rez_spy(self):
        totalReturnGraph(['spy','rez','vnq'],period='10y')

    def test_dea_hasi_spy(self):
        totalReturnGraph(['dea','hasi','spy'])

    def test_real_interest_rates(self):
        totalReturnGraph(['SCHP'])

