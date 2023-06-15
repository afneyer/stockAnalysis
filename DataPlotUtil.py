from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import figure, pyplot as plt
from matplotlib import axes
from pandas import DataFrame

from DataImporter import MyData
from NumUtilities import return_over_number_periods


def plot_sp500_monthly_logscale(df: DataFrame):  # -> Tuple[figure, axes]:
    y = df[MyData.sp500_div_reinvest_month].squeeze().astype(float).to_numpy()
    dates = df.index.values
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

def plot_sp500_monthly_logscale_new(df: DataFrame):  # -> Tuple[figure, axes]:
    y = df[MyData.sp500_div_reinvest_month].squeeze().to_numpy().astype(float)
    x = df.index.values.astype(float)

    # print(np.where(~np.isfinite(x)))
    # print(np.where(~np.isfinite(y)))
    logy = np.log(y)
    p = np.polyfit(x, logy, 1)

    # Convert the polynomial back into an exponential
    a = np.exp(p[1])
    b = p[0]
    y_fitted = a * np.exp(b * x)

    # Using Matplotlib for plotting
    xaxis = df.index.values

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
