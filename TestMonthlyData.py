import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes

import DataPlotUtil
import NumUtilities
from DataFileReader import DataFileReader
from DataImporter import DataImporter, MyData
from NumUtilities import return_over_number_periods


class TestMonthlyData(TestCase):

    def test_sp500_total_return_basic(self):
        di = DataImporter()
        df = di.import_all_series()
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale_new(df)
        plt.show()


    def test_plot_based_on_data_frame(self):
        df = DataFileReader().read_us_market_visualizations()

        # Plot the SP500 total return
        plot = df.plot(y=['SP500+DivMonthly'], figsize=(20, 10), grid=True, logy=True)
        plot.grid('on', which='minor', axis='y')
        plt.show()

    def test_plot_of_earnings_return(self):
        di = DataImporter()
        df = di.import_all_series()
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale_new(df)

        label = "Earnings Reinvested"
        tret = df[MyData.sp500_earnings_growth].squeeze().to_numpy()
        ax.plot(df.index.values, tret, 'b', label=label)

        label = "PE-Ratio"
        pe = df[MyData.sp500_pe_ratio_month].squeeze().to_numpy()
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df.index.values, pe, 'c', label=label)

        fig.set_figheight(7)
        fig.set_figwidth(10)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.show()

    def test_plot_of_earnings_and_dividend_yield(self):
        di = DataImporter()
        df = di.import_all_series()
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale_new(df)

        label = "Earnings Yield"
        ey = df[MyData.sp500_earnings_yield].squeeze().to_numpy()
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df.index.values, ey, 'b', label=label)

        label = "Dividend Yield"
        ey = df[MyData.sp500_div_yield_month].squeeze().to_numpy() / 100.0
        ax2.plot(df.index.values, ey, 'c', label=label)

        fig.set_figheight(7)
        fig.set_figwidth(10)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
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
        plt.legend(loc='upper right')
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

        fig.tight_layout()
        # otherwise the right y-label is slightly clipped
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

    def test_SP500_and_GDP_percent_increase(self):
        di = DataImporter()
        df = di.import_series(MyData.sp500_div_reinvest_month)
        df = di.import_series(MyData.us_gdp_nominal)

        start = df[MyData.us_gdp_nominal].first_valid_index()
        df = df.loc[start:]

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df)
        xaxis = df.index.values

        sp500_percent_inc = df[MyData.sp500_div_reinvest_month].pct_change()
        print(len(df))
        y1 = NumUtilities.moving_average(sp500_percent_inc, 36)
        gdp_percent_inc = df[MyData.us_gdp_nominal].pct_change()
        print(len(df))
        y2 = NumUtilities.moving_average(gdp_percent_inc, 36)
        xaxis = xaxis[36:]

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        label = "SP500 Percent Increase to Previous Month"
        ax2.plot(xaxis, y1, 'b', label=label, linewidth=1.0, linestyle='-', ms=1)

        label = "GDB Percent Increase to Previous Month"
        ax2.plot(xaxis, y2, 'c', label=label, linewidth=1.0, linestyle='-', ms=1)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_figheight(7.5)
        fig.set_figwidth(10)
        plt.legend(loc='upper right')
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()




