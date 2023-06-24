import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

import DataPlotUtil
import NumUtilities
from DataFileReader import DataFileReader
from DataImporter import DataImporter
from MyData import MyData
from NumUtilities import return_over_number_periods


class TestMonthlyData(TestCase):

    def test_sp500_total_return_basic(self):
        di = DataImporter()
        DataPlotUtil.plot_sp500_monthly_logscale(di,MyData.sp500_div_reinvest_month)
        plt.show()

    def test_plot_based_on_data_frame(self):
        df = DataFileReader().read_us_market_visualizations()

        # Plot the SP500 total return
        plot = df.plot(y=['SP500+DivMonthly'], figsize=(20, 10), grid=True, logy=True)
        plot.grid('on', which='minor', axis='y')
        plt.show()

    def test_plot_of_earnings_return(self):
        required_series = [MyData.sp500_earnings_growth, MyData.sp500_pe_ratio_month]
        di = DataImporter()

        df = di.get_selected_series_as_df(required_series)
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di,MyData.sp500_div_reinvest_month)

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
        required_series = [MyData.sp500_earnings_yield, MyData.sp500_div_yield_month]
        di = DataImporter()
        df = di.get_selected_series_as_df(required_series)
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di,MyData.sp500_div_reinvest_month)

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

    @unittest.skip
    def test_monthly_data_with_60_month_return(self):
        df = DataFileReader().read_us_market_visualizations()

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df,'SP500+DivMonthly')

        # Compute n-period return
        n = 1  # three year returns
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

    @unittest.skip
    def test_monthly_data_with_10_year_treasury(self):
        df = DataFileReader().read_us_market_visualizations()

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df,'SP500+DivMonthly')

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

    @unittest.skip
    def test_monthly_data_with_earnings_return(self):
        df = DataFileReader().read_us_market_visualizations()

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df,'SP500+DivMonthly')
        xaxis = df['FullDate'].to_numpy()

        sp = df['SP500Index'].to_numpy()
        ea = df['Earnings'].to_numpy()

        tret = NumUtilities.total_return(sp, ea)

        label = "Earnings Reinvested"
        ax.plot(xaxis, tret, 'b', label=label)

        # Show monthly earnings overstatements
        sp500_div = df['SP500+DivMonthly'].to_numpy()

        sp500_div_increase = np.diff(sp500_div[11:]) / sp500_div[:-12] * 100
        sp500_earnings_inc = np.diff(tret[11:]) / tret[:-12] * 100
        diff_earnings = sp500_earnings_inc - sp500_div_increase

        label = "SP500 Overstated Earnings in % per year"
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        x1 = xaxis[12:]
        ax2.plot(x1, diff_earnings, 'p', label=label, linewidth=1.0, linestyle='-', ms=1)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_figheight(7)
        fig.set_figwidth(20)
        plt.legend(loc='upper right')
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def test_SP500_and_GDP_percent_increase(self):
        di = DataImporter()
        df = di.get_selected_series_as_df([MyData.sp500_div_reinvest_month, MyData.us_gdp_nominal])

        start = df[MyData.us_gdp_nominal].first_valid_index()
        df = df.loc[start:]

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di,MyData.sp500_div_reinvest_month)
        xaxis = df.index.values

        sp500_percent_inc = df[MyData.sp500_div_reinvest_month].pct_change()
        print(len(df))
        n = 1
        y1 = NumUtilities.moving_average(sp500_percent_inc, n)
        gdp_percent_inc = df[MyData.us_gdp_nominal].pct_change()
        print(len(df))
        y2 = NumUtilities.moving_average(gdp_percent_inc, n)
        xaxis = xaxis[n-1:]

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

        plt.xcorr(y1, y2, usevlines=True, normed=True, maxlags=60)
        plt.title("Cross Correlation of Total Monthly Return and 10-Year Treasury")
        plt.show()

        DataPlotUtil.scatter_plot_with_regression_line(y2,y1,2)

        # scatter plot gdp increase

    def test_monthly_data_scatter_plot_with_n_period_return(self):
        df = DataImporter().get_series_as_df(MyData.sp500_div_reinvest_month)

        # fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df)

        # Compute n-period return
        n = 1 # one month returns
        xaxis = df.index.values
        y = df[MyData.sp500_div_reinvest_month].squeeze().to_numpy()
        x1, y1 = return_over_number_periods(n, xaxis, y)

        # y2 next n-month return
        y3 = np.array(y1[n:],dtype=float)
        y4 = np.array(y1[:-n],dtype=float)


        label = str(n) + "-month return based previous n-month return"
        m,b = np.polyfit(y3,y4,1)
        y3min = y3.min()
        y3max = y3.max()

        fig, ax = plt.subplots(figsize=(10,7.5))

        ax.scatter(y3, y4,1)

        plt.xlim([y3min,y3max])
        plt.title(label)

        ax.plot(y3, m*y3+b, label='regression line')

        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.plot(x1, y1, 'b', label=label)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper right')
        plt.savefig("sp500_plus_" + label + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
