import datetime as dt
import unittest

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


# noinspection PyMethodMayBeStatic
class MyTestCase(unittest.TestCase):
    def test_basic_data_frame_and_dictionary(self):
        data = {'Key': ['k1', 'k2', 'k3'], 'Value': ['A', 'B', 'C']}
        df1 = pd.DataFrame(data)
        df1 = df1.set_index('Key')
        print(df1)
        i = df1.loc['k2', 'Value']
        assert i == 'B'

    def test_creation_of_dataframe(self):
        data = ['a', 'b', 'c']
        keys = ['k1', 'k2', 'k3']
        columns = ['c1', 'c2']
        df = pd.DataFrame(data=[keys, data], index=columns).T
        print(df)

    def test_merge_of_dataframes(self):
        data = {'Key': [2, 3, 4], 'Value': ['B', 'C', 'D']}
        df1 = pd.DataFrame(data)
        df1.set_index('Key', inplace=True)
        print(df1)
        data = {'Key': [1, 2, 3, 4, 5], 'Value': ['A', 'B', 'C', 'D', 'E']}
        df2 = pd.DataFrame(data)
        df2.set_index('Key', inplace=True)
        print(df2)
        df = pd.merge(df1, df2, on='Key')
        print(df)
        df = pd.merge(df1, df2, on='Key', how='outer', sort=True)
        print(df)

    def test_cubic_spline_interpolation(self):
        x = [0, 1, 2]
        y = [1, 3, 2]

        # use bc_type = 'natural' add the constraint straight lines at the end point
        # i.e. second derivative is zero
        f = CubicSpline(x, y, bc_type='natural')
        x_new = np.linspace(0, 2, 100)
        y_new = f(x_new)

        plt.figure(figsize=(10, 7.5))
        plt.plot(x_new, y_new, 'b')
        plt.plot(x, y, 'ro')
        plt.title('Cubic Spline Interpolation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def test_cubic_spline_interpolation_financial(self):
        d = pd.date_range(start='1/1/2020', end='1/1/2022', freq='Q')
        y = [1.0, 1.1, 1.2, 1.05, 1.15, 1.25, 1.35, 1.25]

        # use bc_type = 'natural' add the constraint straight lines at the end point
        # i.e. second derivative is zero
        f = CubicSpline(d, y, bc_type='natural')
        d_new = pd.date_range(start='1/1/2020', end='1/1/2022', freq='MS')
        y_new = f(d_new)

        plt.figure(figsize=(10, 7.5))
        plt.plot(d_new, y_new, 'b')
        plt.plot(d, y, 'ro')
        plt.title('Cubic Spline Interpolation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def test_cubic_spline_interpolation_financial_on_dataframe(self):
        d = pd.date_range(start='12/31/2020', end='12/31/2022', freq='Q') + pd.Timedelta(days=1)
        y = [1.0, 1.1, 1.2, 1.05, 1.15, 1.25, 1.35, 1.25, 1.20]

        df = pd.DataFrame({'Date': d, 'Value': y})

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # df.set_index('Date')
        df2 = df.asfreq('MS')
        df_new = df2.interpolate(method='cubicspline')
        df_new.rename(columns={'Value': 'Value_New'}, inplace=True)  # needed for merging
        df_both = pd.merge(df_new, df, on='Date', how='outer', sort=True)

        # use bc_type = 'natural' add the constraint straight lines at the end point
        # i.e. second derivative is zero
        f = CubicSpline(d, y, bc_type='natural')
        d_new = pd.date_range(start='1/1/2020', end='1/1/2022', freq='MS')
        y_new = f(d_new)

        plt.figure(figsize=(10, 7.5))
        df_both.plot(y=['Value', 'Value_New'], style=['ro', 'b-'], use_index=True)
        plt.plot(d_new, y_new, 'b')
        plt.plot(d, y, 'ro')
        plt.title('Cubic Spline Interpolation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def test_reading_data_from_yahoo_SPY(self):
        start = dt.datetime(2000, 1, 1)
        end = dt.date.today()
        all_data = yf.Ticker('SPY')
        t_hist = all_data.history(period='max')
        t_div = all_data.dividends
        t_cap = all_data.capital_gains

        print(t_cap)

    def test_dictionary_with_data_frames(self):
        data = {'Key': ['k1', 'k2', 'k3'], 'Value': ['A', 'B', 'C']}
        df1 = pd.DataFrame(data)
        df1.set_index('Key', inplace=True)
        df2 = pd.DataFrame(data)
        df2.set_index('Key', inplace=True)
        print(df1)
        my_dict = {'key1': df1, 'key2': df2}
        df = my_dict.get('key1')
        my_dict['key3'] = df1
        print(df)

    # apparently there is no way to put dictionaries into dictionaries
    def test_data_frame_with_data_frames(self):
        data = {'Key': ['k1', 'k2', 'k3'], 'Value': ['A', 'B', 'C']}
        df1 = pd.DataFrame(data)
        df1.set_index('Key', inplace=True)
        df2 = pd.DataFrame(data)
        df2.set_index('Key', inplace=True)
        df3 = df2
        print(df1)
        data1 = {'Key': ['row1', 'row2', 'row3'], 'Value': [0.0, 1.1, 2.2]}
        dfofdf = pd.DataFrame(data1)
        dfofdf.set_index('Key', inplace=True)
        # dfofdf['Series'] = np.nan
        dfofdf.at['row1', 'Series'] = 10.0
        dfofdf.at['row2', 'Series'] = df2.copy()
        dfofdf.loc['row3', 'Series'] = pd.DataFrame()
        df = dfofdf.loc['row1', 'Series']
        df4 = dfofdf.at['row3', 'Series']
        print(df)
        print(df4)

    def test_converting_to_numeric(self):
        data = {'Key': ['k1', 'k2', 'k3'], 'Value': [7, 6, 5]}
        df = pd.DataFrame(data)
        df1 = df.copy()
        df1['Value'] = pd.to_numeric(df1['Value'])
        assert df1['Value'].dtype == 'int64'

        s = pd.Series([1, np.nan, 3.0], dtype="float64")
        s1 = pd.to_numeric(s, downcast="float")
        assert s1.dtype == 'float32'
