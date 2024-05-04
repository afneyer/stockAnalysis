import os
from unittest import TestCase

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from numpy.ma.testutils import approx
from pandas import DataFrame

from DataImporter import DataImporter, adjust_dates_to_start_of_month, restore_date_index, \
    change_to_row_index, all_dates_month_start, check_all_dates_daily, check_all_dates_daily_contiguous, print_df
from MyData import MyData as Md


def assert_value(df: DataFrame, df_column: str, date: str, val: float):
    ts = pd.Timestamp(date)
    if approx(df[df_column][ts], val):
        return
    else:
        print('--> Value Error: Target =' + str(val) + ' Actual=' + str(df[df_column][ts]))


class TestDataImporter(TestCase):

    def test_constructor(self):
        di = DataImporter()
        fs_id = Md.sp500_real_price_month
        assert di.get_url(fs_id) == 'MULTPL/SP500_REAL_PRICE_MONTH'
        assert di.get_url_type(fs_id) == Md.quandle
        di.display_series_dictionary()

    def test_get_series_as_df(self):
        # Verify existing series is retrieved
        di = DataImporter()
        df = di.get_series_as_df(Md.sp500_real_price_month)
        df1 = di.get_series_as_df(Md.sp500_real_price_month)
        assert df.equals(df1)

        # Verify that non-existing series are imported on demand (including prerequisites)
        df = di.get_series_as_df(Md.sp500_div_reinvest_month)
        assert Md.sp500_div_reinvest_month in df.columns

    def test_get_series_as_series(self):
        di = DataImporter()
        df = di.get_series_as_df(Md.sp500_real_price_month)
        fs = di.get_series_as_series(Md.sp500_real_price_month)
        assert fs.equals(df.squeeze())

    def test_get_series_as_numpy(self):
        di = DataImporter()
        df = di.get_series_as_df(Md.sp500_real_price_month)
        fsn = di.get_series_as_numpy(Md.sp500_real_price_month)
        assert np.allclose(fsn, df.squeeze().astype(float).to_numpy())

    def test_get_data_from_quandle(self):
        di = DataImporter()
        fs_id = Md.sp500_div_yield_month
        url = di.get_url(fs_id)
        df = di.get_data_from_quandle(url)
        df = df.rename(columns={'Value': fs_id})
        ts = pd.Timestamp('2022-8-31')
        assert approx(df[Md.sp500_div_yield_month][ts], 1.56)
        print(df.head())

    def test_get_data_from_multpl(self):
        cpi_url = 'https://www.multpl.com/cpi/table/by-month'

        di = DataImporter()
        cpi_data = di.get_data_from_multpl_website(cpi_url)
        print(cpi_data.head())

        ts = pd.Timestamp('2022-12-01')
        assert approx(cpi_data['Value'][ts], 296.80)

    def test_get_series_as_df_with_dependencies(self):
        di = DataImporter()
        fs_id = Md.sp500_div_reinvest_month
        df = di.get_series_as_df(fs_id)
        display(df.head())

        # ts = pd.Timestamp('2022-12-01') TODO
        # assert approx(df[fs_id][ts], 296.80)

    def test_adjust_dates_quandle(self):
        di = DataImporter()
        col_nam = Md.sp500_div_yield_month
        url = di.get_url(col_nam)
        df = di.get_data_from_quandle(url)
        print(df.head)
        df = adjust_dates_to_start_of_month(df)
        assert all_dates_month_start(df)
        print(df.head)

    def test_adjust_dates_to_start_of_month(self):
        rng1 = pd.date_range('2015-03-30', periods=10, freq='SM')
        rng2 = pd.date_range('2015-05-01', periods=3, freq='MS')
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

    def test_fred_with_gdp(self):
        di = DataImporter()
        df = di.get_series_as_df(Md.us_gdp_nominal)
        print(df)

    def test_get_url(self):
        ser_id = Md.sp500_div_yield_month
        di = DataImporter()
        url = di.get_url(ser_id)
        assert url == 'MULTPL/SP500_DIV_YIELD_MONTH'

    def test_get_url_type(self):
        ser_id = Md.sp500_div_yield_month
        di = DataImporter()
        url = di.get_url_type(ser_id)
        assert url == Md.quandle

    # Test each individual data series separately
    def test_sp500_pe_ratio_month(self):
        series_id = Md.sp500_pe_ratio_month
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 22.65)

    def test_sp500_div_yield_month(self):
        series_id = Md.sp500_div_yield_month
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 0.016)

    def test_sp500_real_price_month(self):
        series_id = Md.sp500_real_price_month
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 3912.38)

    def test_cpi_urban_month(self):
        series_id = Md.cpi_urban_month
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 296.80)

    def test_ten_year_treasury_month(self):
        series_id = Md.ten_year_treasury_month
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 0.0362)

    def test_us_gdp_nominal_month(self):
        series_id = Md.us_gdp_nominal
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        ts = pd.Timestamp('2021-10-01')
        value = df[Md.us_gdp_nominal][ts]
        assert_value(df, series_id, '2021-10-01', 24249.121)
        assert value > 23202

    def test_sp500_div_reinvest_month(self):
        series_id = Md.sp500_div_reinvest_month
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 2552332)

    def test_sp500_earnings_growth(self):
        series_id = Md.sp500_earnings_growth
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 269655.4)

    def test_sp500_earnings_yield(self):
        series_id = Md.sp500_earnings_yield
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        assert all_dates_month_start(df)
        assert_value(df, series_id, '2022-12-01', 0.04415)

    def test_sp500_ten_year_minus_two_year_treasury_cm(self):
        series_id = Md.ten_year_minus_two_year
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)
        # assert all_dates_month_start(df)
        # assert_value(df, series_id, '2022-12-01', 0.04415)

    def test_int_treasury(self):
        int_series = [Md.int_one_month_cm, Md.int_three_month_cm, Md.int_six_month_cm,
                      Md.int_one_year_cm, Md.int_two_year_cm, Md.int_five_year_cm,
                      Md.int_ten_year_cm, Md.int_thirty_year_cm]
        df = DataImporter().get_selected_series_as_df(int_series)

        df.to_csv('./printouts/interest_rates.csv')
        print(df.head)

    def test_sp500_div_reinvest_day(self):
        series_id = Md.sp500_div_reinvest_day
        df = DataImporter().get_series_as_df(series_id)
        print(df.head)

    # test al data importer functions

    def test_all_dates_daily(self):
        df = DataImporter().get_series_as_df(Md.int_two_year_cm)
        assert not check_all_dates_daily(df)
        df = df.asfreq('D')
        df = df.interpolate(method='cubicspline')
        with pd.option_context('display.max_rows', None):
            print(df)
        assert check_all_dates_daily(df)

    def test_all_values_contiguous(self):
        rng = pd.date_range('2015-04-30', periods=6, freq='D')
        df = pd.DataFrame({'Date': rng, 'Value': np.arange(0.0, len(rng), 1.0)})
        df.set_index('Date', inplace=True)
        print(df)
        print(df.head)
        assert check_all_dates_daily_contiguous(df)
        rng1 = pd.date_range('2015-04-30', periods=3, freq='D')
        rng2 = pd.date_range('2015-05-04', periods=3, freq='D')
        rng = rng1.union(rng2)
        df = pd.DataFrame({'Date': rng, 'Value': np.arange(0.0, len(rng), 1.0)})
        df.set_index('Date', inplace=True)
        print(df)
        print(df.head)
        assert check_all_dates_daily_contiguous(df)

    def test_all_data_series(self):
        output_folder = "output"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        output_file = os.path.join(output_folder, "Data Series Overview.txt")
        out_file = open(output_file, "w")

        all_series = [row[0] for row in Md.urls]
        for i in range(0, len(all_series)):
            s_id = all_series[i]
            df = DataImporter().get_series_as_df(s_id)
            out_file.write("Series Name:              " + Md.urls[i][0] + "\n")
            out_file.write("Series Data Source:       " + Md.urls[i][1] + "\n")
            out_file.write("Series Url:               " + str(Md.urls[i][2]) + "\n")
            out_file.write("Series Number:            " + str(i) + "\n")

            series = df[s_id].squeeze()
            d = series.index[0]
            dv = series.first_valid_index()
            out_file.write("Series First Index:       " + str(d) + "   First Valid: " + str(dv) + "\n")
            d = series.index[len(series)-1]
            dv = series.last_valid_index()
            out_file.write("Series Last Index:        " + str(d) + "   Last Valid:  " + str(dv) + "\n")
            out_file.write("----------------------------------------------------------------------\n\n")

        out_file.close()

    def test_specific_series(self):
        fs_id = Md.mult_eco_us_population
        di = DataImporter()
        df = di.get_series_as_df(fs_id)
        print_df(df)

