from unittest import TestCase

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from numpy.ma.testutils import approx
from pandas import DataFrame

from DataImporter import DataImporter, MyData, adjust_dates_to_start_of_month, restore_date_index, \
    change_to_row_index


def all_dates_month_start(df):
    return False not in (pd.to_datetime(df.index.values).day == 1)


def assert_value(df: DataFrame, df_column: str, date: str, val: float):
    ts = pd.Timestamp(date)
    assert approx(df[df_column][ts], val)


class TestDataImporter(TestCase):

    def test_constructor(self):
        di = DataImporter()
        fs_id = MyData.sp500_real_price_month
        assert di.get_url(fs_id) == 'MULTPL/SP500_REAL_PRICE_MONTH'
        assert di.get_url_type(fs_id) == MyData.quandle
        di.display_series_dictionary()

    def test_get_data_from_quandle(self):
        di = DataImporter()
        fs_id = MyData.sp500_div_yield_month
        url = di.get_url(fs_id)
        df = di.get_data_from_quandle(url)
        df = df.rename(columns={'Value': fs_id})
        ts = pd.Timestamp('2022-8-31')
        assert approx(df[MyData.sp500_div_yield_month][ts], 1.56)
        print(df.head())

    def test_get_data_from_multpl(self):
        cpi_url = 'https://www.multpl.com/cpi/table/by-month'

        di = DataImporter()
        cpi_data = di.get_data_from_multpl_website(cpi_url)
        print(cpi_data.head())

        ts = pd.Timestamp('2022-12-01')
        assert approx(cpi_data['Value'][ts], 296.80)



    def test_import_series_with_dependencies(self):
        di = DataImporter()
        fs_id = MyData.sp500_div_reinvest_month
        df = di.import_series(fs_id)
        display(df.head())

        # ts = pd.Timestamp('2022-12-01')
        # assert approx(df[fs_id][ts], 296.80)
    def test_import_all_data(self):
        di = DataImporter()
        di.import_all_series()

    def test_adjust_dates_quandle(self):
        di = DataImporter()
        url = di.data_dict.loc[MyData.sp500_div_yield_month]['url']
        df = di.get_data_from_quandle(url)
        print(df.head)
        df = adjust_dates_to_start_of_month(df)
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
        assert approx(df['Value'], [0.0, 3.0, 6.0, 9.0, 11.0], 0.001).all
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
        print(df)  # add validate TODO
        change_to_row_index(df)
        print(df)  # add validation TODO

    def test_real_gdp_quandle(self):
        di = DataImporter()
        url = di.data_dict.loc[MyData.us_gdp_nominal]['url']
        df = di.get_data_from_quandle(url)
        df2 = df.asfreq('MS')
        df_new = df2.interpolate(method='cubicspline')
        print(df_new.head)
        # change_to_row_index(df)
        # df = adjust_dates_to_start_of_month(df)
        # restore_date_index(df)
        print(df.head)
        # col_nam = MyData.sp500_div_yield_month
        # df = df.rename(columns={'Value': col_nam})

    def test_div_reinvest(self):
        di = DataImporter()
        url = id.data_dict_.log[MyData.sp500_div_reinvest_month]

    def test_get_url(self):
        ser_id = MyData.sp500_div_yield_month
        di = DataImporter()
        url = di.get_url(ser_id)
        assert url == 'MULTPL/SP500_DIV_YIELD_MONTH'

    def test_get_url_type(self):
        ser_id = MyData.sp500_div_yield_month
        di = DataImporter()
        url = di.get_url_type(ser_id)
        assert url == MyData.quandle

    # Test each individual data series separately
    def test_sp500_pe_ratio_month(self):
        series_id = MyData.sp500_pe_ratio_month
        df = DataImporter().import_series(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df,series_id,'2022-12-01',22.65)

    def test_sp500_div_yield_month(self):
        series_id = MyData.sp500_div_yield_month
        df = DataImporter().import_series(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 0.016)

    def test_sp500_real_price_month(self):
        series_id = MyData.sp500_real_price_month
        df = DataImporter().import_series(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 3912.38)

    def test_cpi_urban_month(self):
        series_id = MyData.cpi_urban_month
        df = DataImporter().import_series(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 296.80)

    def test_ten_year_treasury_month(self):
        series_id = MyData.ten_year_treasury_month
        df = DataImporter().import_series(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 0.0362)

    def test_us_gdp_nominal_month(self):
        series_id = MyData.us_gdp_nominal
        df = DataImporter().import_series(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2021-10-01', 23992.355000)
        ts = pd.Timestamp('2021-09-01')
        value = df[MyData.us_gdp_nominal][ts]
        assert value < 23992
        assert value > 23202

    def test_sp500_div_reinvest_month(self):
        di = DataImporter()
        df = di.import_series(MyData.sp500_div_reinvest_month)
        print(df.head)
        assert all_dates_month_start(df)




