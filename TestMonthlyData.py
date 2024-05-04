from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

import DataPlotUtil
import NumUtilities
from DataImporter import DataImporter
from MyData import MyData
from NumUtilities import return_over_number_periods


def save_figure(label: str, plot):
    plot.savefig('./plots/' + 'sp500_plus_' + label + '.pdf', format='pdf', bbox_inches="tight")


def cross_corr(s1, s2, window=50, max_lags=10):
    if len(s1) != len(s2):
        raise (ValueError('S1 and S2 have different lengths'))
    if max_lags > len(s1):
        raise (ValueError('max_lags has to be less than the length of s1 // 2 i.e.' + str(len(s1))))
    if window > (len(s1) - max_lags) // 2:
        raise (ValueError('window has to be smaller than the length of s1 // 2 i.e.' + str(len(s1) - max_lags // 2)))

    result = np.zeros(2 * max_lags + 1)
    j = 0
    # compute the window range for x
    middle = len(s1) // 2
    w1 = middle - window
    w2 = middle + window
    x = s1[w1:w2]
    for i in range(-max_lags, +max_lags, 1):
        y = s2[w1 + i:w2 + i]
        print(x)
        print(y)
        # least square error produces no results (same for all values)
        # z = np.sum((x-y)**2)
        z = np.sum(x * y)
        result[j] = z
        print(z)
        print(i)
        print(j)
        j += 1
    return result


class TestMonthlyData(TestCase):

    def test_sp500_total_return_basic(self):
        di = DataImporter()
        DataPlotUtil.plot_sp500_monthly_logscale(di, MyData.sp500_div_reinvest_month)
        plt.show()

    def test_plot_of_earnings_return(self):
        required_series = [MyData.sp500_earnings_growth, MyData.sp500_earnings_yield, MyData.sp500_pe_ratio_month,
                           MyData.sp500_div_yield_month,
                           MyData.sp500_real_price_month]
        di = DataImporter()

        df = di.get_selected_series_as_df(required_series)
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di, MyData.sp500_div_reinvest_month)

        label = "Earnings Reinvested"
        tret = df[MyData.sp500_earnings_growth].squeeze().to_numpy()
        ax.plot(df.index.values, tret, 'b', label=label)

        label = "S&P Stock Price"
        tret = df[MyData.sp500_real_price_month].squeeze().to_numpy()
        ax.plot(df.index.values, tret, 'm', label=label)

        label = "Earnings Yield"
        ey = df[MyData.sp500_earnings_yield].squeeze().to_numpy()
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df.index.values, ey, 'c', label=label)

        label = "Dividend Yield"
        dy = df[
            MyData.sp500_div_yield_month].squeeze().to_numpy()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df.index.values, dy, 'y', label=label)

        fig.set_figheight(7)
        fig.set_figwidth(10)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.show()

    def test_plot_of_earnings_and_dividend_yield(self):
        required_series = [MyData.sp500_earnings_yield, MyData.sp500_div_yield_month, MyData.sp500_pe_ratio_month]
        di = DataImporter()
        df = di.get_selected_series_as_df(required_series)
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di, MyData.sp500_div_reinvest_month)

        label = "Earnings Yield"
        ey = df[MyData.sp500_earnings_yield].squeeze().to_numpy()
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df.index.values, ey, 'b', label=label)

        label = "Dividend Yield"
        ey = df[MyData.sp500_div_yield_month].squeeze().to_numpy()
        ax2.plot(df.index.values, ey, 'c', label=label)

        label = "Inverse PE-Ratio"
        pe = df[MyData.sp500_pe_ratio_month].squeeze().to_numpy()
        ey = np.reciprocal(pe)
        ax2.plot(df.index.values, ey, 'r', label=label)

        fig.set_figheight(7)
        fig.set_figwidth(10)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.show()

    def test_SP500_and_GDP_percent_increase(self):
        di = DataImporter()
        df = di.get_selected_series_as_df([MyData.sp500_div_reinvest_month, MyData.us_gdp_nominal])

        start = df[MyData.us_gdp_nominal].first_valid_index()
        df = df.loc[start:]

        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di, MyData.sp500_div_reinvest_month)
        xaxis = df.index.values

        sp500_percent_inc = df[MyData.sp500_div_reinvest_month].pct_change()
        print(len(df))
        n = 1
        y1 = NumUtilities.moving_average(sp500_percent_inc, n)
        gdp_percent_inc = df[MyData.us_gdp_nominal].pct_change()
        print(len(df))
        y2 = NumUtilities.moving_average(gdp_percent_inc, n)
        xaxis = xaxis[n - 1:]

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        label = "SP500 Percent Increase to Previous Month"
        ax2.plot(xaxis, y1, 'b', label=label, linewidth=1.0, linestyle='-', ms=1)

        label = "GDB Percent Increase to Previous Month"
        ax2.plot(xaxis, y2, 'c', label=label, linewidth=1.0, linestyle='-', ms=1)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_figheight(7.5)
        fig.set_figwidth(10)
        plt.legend(loc='upper right')
        save_figure(label, plt)
        plt.show()

        plt.xcorr(y1, y2, usevlines=True, normed=True, maxlags=60)
        plt.title("Cross Correlation of Total Monthly Return and 10-Year Treasury")
        plt.show()

        DataPlotUtil.scatter_plot_with_regression_line(y2, y1, 2)

        # scatter plot gdp increase

    def test_monthly_data_scatter_plot_with_n_period_return(self):
        df = DataImporter().get_series_as_df(MyData.sp500_div_reinvest_month)

        # fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(df)

        # Compute n-period return
        n = 1  # one month returns
        xaxis = df.index.values
        y = df[MyData.sp500_div_reinvest_month].squeeze().to_numpy()
        x1, y1 = return_over_number_periods(n, xaxis, y)

        # y2 next n-month return
        y3 = np.array(y1[n:], dtype=float)
        y4 = np.array(y1[:-n], dtype=float)

        label = str(n) + "-month return based previous n-month return"
        res = np.polyfit(y3, y4, 1)
        m = res[0]
        b = res[1]
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
        save_figure(label, plt)
        plt.show()

    def test_cross_correlation_between_indices(self):
        s1 = MyData.sp500_div_reinvest_month
        s2 = MyData.mult_eco_consumer_price_index_cpi

        df = DataImporter().get_selected_series_as_df([s1, s2])
        sp500_percent_inc = df[MyData.sp500_div_reinvest_month].pct_change()

        os = df[s2]

        y = sp500_percent_inc.to_numpy()
        # remove first element
        y = y[1:]

        x = os.to_numpy()
        # remove last element
        x = x[0:len(x) - 1]

        # Change the length of the series
        num_period = 120
        x = x[len(x) - num_period:len(x)]
        y = y[len(y) - num_period:]

        plt.scatter(x, y)

        plt.show()

        max_lags = 60

        plt.subplots(figsize=(10, 7.5))
        plt.xcorr(x, y, usevlines=True, normed=True, maxlags=max_lags)

        plt.title("Cross Correlation of SP500 Total Monthly Return and " + s2)
        plt.legend()
        plt.show()

        z = cross_corr(x, y, window=150, max_lags=50)

        fig, ax = plt.subplots()

        label = "Least Squares Error Correlation"
        x = np.linspace(0, len(z) - 1, len(z))
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(x, z, 'b', label=label)
        plt.show()

        plt.acorr(y, usevlines=True, normed=True, maxlags=max_lags)
        # plt.show()

    def test_cross_correlation_for_forecast(self):
        # x predicts y
        time = np.arange(0, 90, 1)
        x = np.logspace(0, 0.5, 100)

        np.random.seed(10)
        r = np.random.normal(loc=0.0, scale=0.05, size=100)
        x = x + r * x

        y = x[0:90]
        x = x[10:]
        plt.plot(time, x)
        plt.plot(time, y)
        plt.show()

        max_lags = 80
        plt.xcorr(x, y, usevlines=True, normed=True, maxlags=max_lags)
        plt.show()

        x = np.diff(x) / x[0:len(x) - 1]
        y = np.diff(y) / y[0:len(y) - 1]
        time = time[0:len(time) - 1]

        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.title("X predicts Y by 10 days")
        plt.legend(loc='upper left')
        plt.show()

        max_lags = 80
        plt.xcorr(x, y, usevlines=True, normed=True, maxlags=max_lags)
        plt.show()
