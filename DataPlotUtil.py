from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import figure, pyplot as plt
from matplotlib import axes
from pandas import DataFrame, Series

from DataImporter import DataImporter
from MyData import MyData
from NumUtilities import return_over_number_periods


def plot_sp500_monthly_logscale(di: DataImporter, fs_id):  # -> Tuple[figure, axes]:
    fs: Series = di.get_series_as_series(fs_id)
    y = di.get_series_as_numpy(fs_id).astype(float)
    dates = fs.index.values
    x = dates.astype(float)

    # print(np.where(~np.isfinite(x)))
    # print(np.where(~np.isfinite(y)))
    logy = np.log(y)
    p = np.polyfit(x, logy, 1)

    # Convert the polynomial back into an exponential
    a = np.exp(p[1])
    b = p[0]
    y_fitted = a * np.exp(b * x)

    # Using Matplotlib for plotting
    xaxis = dates

    # set the graph parameters
    ax1: axes
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.grid(True, which='minor')

    title = "SP Index with Dividends Reinvested"
    # plt.title(title) # Todo does not work
    ax1.set_title(title)
    ax1.semilogy(xaxis, y, 'r', label=title)
    ax1.semilogy(xaxis, y_fitted, 'g', label="Exponential Fit")
    plt.legend()

    return fig, ax1

