import unittest

import numpy
import numpy as np
from numpy.ma.testutils import approx
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from DfUtil import scale_to_start_value
from NumUtilities import return_over_number_periods, yield_return, moving_average, first_non_nan, replace_leading_nan


def decimal_almost_equal(desired: DataFrame, actual: DataFrame, decimal: int):
    return (abs(desired - actual) < (0.5 * 10.0 ** -decimal)).all()


class MyTestCase(unittest.TestCase):
    def test_scale_to_start_value(self):
        df = DataFrame({'col1': [2.5, 3.0, 3.5], 'col2': [10.0, 11.0, 12.0]})
        df_target = DataFrame({'col1': [1.0, 1.2, 1.4], 'col2': [1.0, 1.1, 1.2]})
        scale_to_start_value(df)
        assert_frame_equal(df,df_target, check_exact=False)

