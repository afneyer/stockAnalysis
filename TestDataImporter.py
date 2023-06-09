from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.ma.testutils import approx

from DataImporter import DataImporter, MyData, adjust_dates_to_start_of_month, restore_date_index, \
    change_to_row_index


class TestDataImporter(TestCase):

    def test_constructor(self):
        di = DataImporter()
        df = di.data_dict
        print(df.head)
        assert df.index.name == 'series'
        row = df.loc[MyData.sp500_real_price_month]
        assert row['type'] == 'quandle'
        assert row['url'] == 'MULTPL/SP500_REAL_PRICE_MONTH'

    def test_get_data_from_quandle(self):
        di = DataImporter()
        url = di.data_dict.loc[MyData.sp500_div_yield_month]['url']
        df = di.get_data_from_quandle(url)
        col_nam = MyData.sp500_div_yield_month
        df = df.rename(columns={'Value': col_nam})
        ts = pd.Timestamp('2022-8-31')
        assert approx(df[MyData.sp500_div_yield_month][ts], 1.56)
        print(df.head())

    def test_get_data_from_multpl(self):
        cpi_url = 'https://www.multpl.com/cpi/table/by-month'

        di = DataImporter()
        cpi_data = di.get_data_from_multpl_website(cpi_url)

        print(cpi_data.head())
        # write tests

    def test_import_my_data(self):
        di = DataImporter()
        di.import_my_data()

    def test_adjust_dates_quandle(self):
        di = DataImporter()
        url = di.data_dict.loc[MyData.sp500_real_price_month]['url']
        df = di.get_data_from_quandle(url)
        print(df.head)
        change_to_row_index(df)
        df = adjust_dates_to_start_of_month(df)
        restore_date_index(df)
        print(df.head)
        col_nam = MyData.sp500_div_yield_month
        df = df.rename(columns={'Value': col_nam})
        # ts = pd.Timestamp('2022-8-31')
        # assert approx(df[MyData.sp500_div_yield_month][ts], 1.56)

    def test_adjust_dates_to_start_of_month(self):
        rng1 = pd.date_range('2015-03-30', periods=10, freq='SM')
        rng2 = pd.date_range('2015-05-01', periods=3, freq='MS')
        rng3 = pd.date_range
        rng = rng1.union(rng2)
        df = pd.DataFrame({'Date': rng, 'Value': np.arange(0.0, len(rng), 1.0)})
        print(df)
        df = adjust_dates_to_start_of_month(df)
        print(df)
        assert len(df) == 5
        assert (df.index == [0, 3, 6, 9, 11]).all
        assert approx (df['Value'],[0.0,3.0,6.0,9.0,11.0], 0.001).all
        # check that every date is a start month date
        assert df['Date'].apply(lambda x: x.is_month_start).all()

    def test_restore_date_index_and_change_to_row_index(self):
        rng1 = pd.date_range('2015-04-30', periods=6, freq='M')
        rng2 = pd.date_range('2015-05-01', periods=2, freq='MS')
        rng = rng1.union(rng2)
        df = pd.DataFrame({'Date': rng, 'Value': np.arange(0.0, len(rng), 1.0)})
        df.index.name = 'RowNum'
        print(df)
        restore_date_index(df)
        print(df) # add validate TODO
        change_to_row_index(df)
        print(df) # add validation TODO
