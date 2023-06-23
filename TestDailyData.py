import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

import DataPlotUtil
import NumUtilities
from DataFileReader import DataFileReader
from DataImporter import DataImporter
from MyData import MyData
from NumUtilities import return_over_number_periods


class TestDailyData(TestCase):

    def test_sp500_total_return_basic(self):
        di = DataImporter()
        DataPlotUtil.plot_sp500_monthly_logscale(di, MyData.sp500_div_reinvest_day)
        plt.show()

    def test_monthly_data_scatter_plot_with_n_period_return(self):
        df = DataImporter().get_series_as_df(MyData.sp500_div_reinvest_day)

        # Compute n-period return
        n = 1# one month returns
        xaxis = df.index.values
        y = df[MyData.sp500_div_reinvest_day].squeeze().to_numpy()
        x1, y1 = return_over_number_periods(n, xaxis, y)

        # y2 next n-month return
        y3 = np.array(y1[n:], dtype=float)
        y4 = np.array(y1[:-n], dtype=float)

        label = str(n) + "-day return based previous n-day return"
        m, b = np.polyfit(y3, y4, 1)
        y3min = y3.min()
        y3max = y3.max()

        fig, ax = plt.subplots(figsize=(10, 7.5))

        ax.scatter(y3, y4, 1)

        plt.xlim([y3min, y3max])
        plt.title(label)

        ax.plot(y3, m * y3 + b, label='regression line')

        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.plot(x1, y1, 'b', label=label)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper right')
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def test_histogram_of_daily_returns(self):
        df = DataImporter().get_series_as_df(MyData.sp500_div_reinvest_day)

        # Compute n-period return
        n = 731# one month returns
        xaxis = df.index.values
        y = df[MyData.sp500_div_reinvest_day].squeeze().to_numpy()
        x1, y1 = return_over_number_periods(n, xaxis, y)

        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')
        ax.minorticks_on()
        plt.hist(y1,bins=500)

        fig_label = "Histogram of " + str(n) + "-day rolling returns"
        plt.title(fig_label)
        plt.savefig(fig_label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
