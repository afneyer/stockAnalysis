import datetime
from unittest import TestCase

import pandas as pd
from numpy.ma.testutils import approx

from DataImporter import DataImporter, MyData


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
        assert approx( df[MyData.sp500_div_yield_month][ts], 1.56)

    def test_get_data_from_multpl(self):
        pe_url = 'http://www.multpl.com/table?f=m'
        price_url = 'http://www.multpl.com/s-p-500-historical-prices/table/by-month'
        cpi_url = 'https://www.multpl.com/cpi/table/by-month'

        di = DataImporter()
        cpi_data = di.get_data_from_multpl_website(cpi_url)

        print(cpi_data.head())
        # write tests
