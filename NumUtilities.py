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
    p1 = np.roll(price, 1)
    tot_ret_per_period = (price + div) / p1

    # Replace the first row's NA with 1.0
    tot_ret_per_period[0] = price[0]

    # Calculate the cumulative Total Return.
    tot_ret = np.nancumprod(tot_ret_per_period)

    return tot_ret


def yield_return(initial: float, yield_percent: ndarray) -> numpy:
    y_ret = np.zeros_like(yield_percent)
    for (yield_percent,idx) in yield_percent:
        if idx == 0:
            y_ret[idx] = initial
        else:
            y_ret[idx] = y_ret[idx - 1] * (1.0 + yield_percent[idx])
        idx += 1

    return y_ret


def moving_average(x: ndarray, n: int) -> numpy:
    if n == 0:
        return x
    x1 = replace_leading_nan(x, 0.0)

    # use numpy.cumsum
    x1 = np.insert(x1, 0, 0)
    cumsum_vec = np.cumsum(x1)
    y = (cumsum_vec[n:] - cumsum_vec[:-n]) / n
    return y


def first_non_nan(x: ndarray) -> int:
    non_nan_indices = np.where(~np.isnan(x))[0]
    f_non_nan = non_nan_indices[0]
    return f_non_nan


def replace_leading_nan(x: ndarray, value: float = 0.0) -> ndarray:
    f_non_nan = first_non_nan(x)
    if f_non_nan > 0:
        x1 = np.full(f_non_nan, value)
    else:
        return x
    x2 = x[f_non_nan:]
    y = numpy.concatenate((x1, x2))
    return y
