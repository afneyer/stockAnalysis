import unittest
from datetime import timedelta
from unittest import TestCase

from matplotlib import pyplot as plt

import DataPlotUtil
from DataImporter import DataImporter, adjust_dates_to_start_of_month
from MyData import MyData


class MyTestCase(TestCase):
    def test_yahoo_spy_daily(self):
        di = DataImporter()
        df = di.get_series_as_df(MyData.yahoo_spy_div_reinvest)
        fig, ax = DataPlotUtil.plot_logscale_with_exponential_fit(df[MyData.yahoo_spy_div_reinvest])
        plt.show()

    def test_yahoo_spy_daily_to_monthly(self):
        di = DataImporter()
        df = di.get_series_as_df(MyData.yahoo_spy_close)
        df1 = df.copy()
        adjust_dates_to_start_of_month(df1)
        s = df[MyData.yahoo_spy_close]
        # series with start of the month dates
        s1 = df1[MyData.yahoo_spy_close]
        for date in df1.index:
            if date in df.index:
                assert s1.loc[date] == s.loc[date]
            else:
                days = 1
                date1 = date + timedelta(days=-days)
                while date1 not in df.index and days < 7:
                    days += 1
                    date1 = date + timedelta(days=-days)

                if date1 in df.index:
                    if abs(s1.loc[date]- s.loc[date1]) > 1e-6:
                        print(date, s1.loc[date],date1, s.loc[date1])
                        assert False
                        # self.assertAlmostEqual(s1.loc[date],s.loc[date1],6)
                else:
                    print("Error: Cannot find a date in the original series within 7 days of " + str(date))
                    assert False



