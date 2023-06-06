import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes

import DataPlotUtil
import NumUtilities
from DataFileReader import DataFileReader
from NumUtilities import return_over_number_periods


class TestMonthlyData(TestCase):

    def test_plot_based_on_data_frame(self):
        df = DataFileReader().read_us_market_visualizations()

        # Plot the SP500 total return
        plot = df.plot(y=['SP500+DivMonthly'], figsize=(20, 10), grid=True, logy=True)
        plot.grid('on', which='minor', axis='y')
        plt.show()

    def test_monthly_data_with_60_month_return(self):
        df = DataFileReader().read_us_market_visualizations()

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df)

        # Compute n-period return
        n = 60  # three year returns
        xaxis = df['FullDate'].to_numpy()
        y = df['SP500+DivMonthly'].squeeze().to_numpy()
        x1, y1 = return_over_number_periods(n, xaxis, y)

        label = str(n) + "-month return"
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(x1, y1, 'b', label=label)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def test_monthly_data_with_10_year_treasury(self):
        df = DataFileReader().read_us_market_visualizations()

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df)

        xaxis = df['FullDate'].to_numpy()
        y = df['10-YearTreasury'].squeeze().to_numpy()

        label = "10-Year Treasury Yield"
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(xaxis, y, 'b', label=label)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper right')
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def test_monthly_data_with_earnings_return(self):
        df = DataFileReader().read_us_market_visualizations()

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df)
        xaxis = df['FullDate'].to_numpy()

        sp = df['SP500Index'].to_numpy()
        ea = df['Earnings'].to_numpy()

        tret = NumUtilities.total_return(sp,ea)

        label = "Earnings Reinvested"
        ax.plot(xaxis, tret, 'b', label=label)

        # Show monthly earnings overstatements
        sp500Div = df['SP500+DivMonthly'].to_numpy()

        sp500DivIncrease = np.diff(sp500Div[11:]) / sp500Div[:-12] * 100
        sp500EarningsInc = np.diff(tret[11:]) / tret[:-12] * 100
        diffEarnings = sp500EarningsInc - sp500DivIncrease

        label = "SP500 Overstated Earnings in % per year"
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        x1 = xaxis[12:]
        ax2.plot(x1, diffEarnings, 'p', label=label, linewidth=1.0, linestyle='-', ms=1)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_figheight(7)
        fig.set_figwidth(20)
        plt.legend(loc='upper right')
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()



'''
    s1 = df['TotalMonthlyReturn'].squeeze()
    s2 = df['10-YearTreasury'].squeeze()

    s1 = s1[1:1800]
    s2 = s2[1:1800]
    x = s1.to_numpy()
    y = s2.to_numpy()

    # Scatter plot of Sp+Div Monthly versus Treasury Yield
    plt.scatter(x, y)
    plt.show()


    # Cross Correlation SP500+Total Return and Treasuries
    plt.xcorr(x, y, usevlines=True, normed=True, maxlags=800)
    plt.title("Cross Correlation of Total Monthly Return and 10-Year Treasury")
    plt.show()

    plt.acorr(x, usevlines=True, normed=True, maxlags=800)
    plt.show()

    plt.acorr(y, usevlines=True, normed=True, maxlags=800)
    plt.show()
'''

'''
if __name__ == '__main__':
    unittest.main()
'''
