from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import figure, pyplot as plt
from matplotlib import axes
from pandas import DataFrame, Series
from pandas.core import series

from DataImporter import DataImporter
from MyData import MyData
from NumUtilities import return_over_number_periods


def plot_logscale_with_exponential_fit(ser: Series):  # -> Tuple[figure, axes]:
    fs: Series = ser
    y = ser.to_numpy(dtype=float)
    dates = fs.index.values.astype(float)
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

def scatter_plot_with_regression_line(x,y,y_delay=0, descript=''):
    # y next n-month return
    y1 = np.array(y[y_delay:], dtype=float)
    x1 = np.array(x[:len(x)-y_delay], dtype=float)

    # label = str(n) + "-day return based previous n-day return"
    m, b = np.polyfit(x1, y1, 1)
    xmin = x1.min()
    xmax = x1.max()

    fig, ax = plt.subplots(figsize=(10, 7.5))

    ax.scatter(x1, y1, 10)

    plt.xlim([xmin, xmax])
    plt.title('Scatterplot of ' + descript + ' with y Delayed by ' + str(y_delay) + ' Periods')

    ax.plot(x1, m * x1 + b, label='regression line')
    plt.show()

