from typing import Tuple

import numpy as np
from numpy import ndarray
import numpy


def return_over_number_periods(n: int, xin: ndarray, yin: ndarray) -> Tuple[ndarray, ndarray]:
    # shift the yin array
    yin1 = yin[0:yin.size - n]
    yin2 = yin[n:]
    xout = xin[0:xin.size - n]
    yout = yin2 / yin1 - 1.0
    return xout, yout


def total_return(price: ndarray, div: ndarray) -> numpy:
    """
    Calculate the "Total Return" of a stock when dividends are
    reinvested in the stock.

    The formula is:

    Total_Return[t] = Total_Return[t-1] * (Dividend[t] + Share_Price[t]) / Share_Price[t-1]
    """

    # Calculate the daily Total Return.
    p1 = np.roll(price,1)
    tot_ret_per_period = (price+div) / p1

    # Replace the first row's NA with 1.0
    tot_ret_per_period[0] = price[0]

    # Calculate the cumulative Total Return.
    tot_ret = tot_ret_per_period.cumprod()



    return tot_ret
