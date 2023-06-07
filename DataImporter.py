import quandl

import pandas as pd
import requests
from pandas import DataFrame, Series
from lxml import html as HTMLParser


class MyData:
    # data types
    quandle = 'quandle'
    multpl = 'multpl'

    # data columns
    sp500_pe_ratio_month = 'SP500_PE_Ratio_Month'
    sp500_div_yield_month = 'SP500_Div_Yield_Month'
    sp500_real_price_month = 'SP500_Real_Price_Month'
    cpi_urban_month = 'CPI_Urban_Month'

    urls = [
        [sp500_pe_ratio_month, quandle, 'MULTPL/SP500_PE_RATIO_MONTH'],
        [sp500_div_yield_month, quandle, 'MULTPL/SP500_DIV_YIELD_MONTH'],
        [sp500_real_price_month, quandle, 'MULTPL/SP500_REAL_PRICE_MONTH'],
        [cpi_urban_month, multpl, 'https://www.multpl.com/cpi/table/by-month']
    ]


class DataImporter:

    data_dict: DataFrame
    # data columns
    def __init__(self):
        self.data_dict = pd.DataFrame(MyData.urls, columns=['series', 'type', 'url'])
        self.data_dict = self.data_dict.set_index('series')

    def import_us_market(self) -> DataFrame:
        df: DataFrame
        df = quandl.get('MULTPL/SP500_PE_RATIO_MONTH')
        x = df['Value']

        return df

    # Quandl API Key:  iYippvXT8yzS7xK47yoh

    def get_data_from_quandle(self,url):
        df = quandl.get(url)
        return df

    def get_data_from_multpl_website(self, url):
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36'
            )
        }
        res = requests.get(url, headers=headers)
        parsed = HTMLParser.fromstring(res.content.decode('utf-8'))
        tr_elems = parsed.cssselect('#datatable tr')
        raw_data = [[td.text.strip() for td in tr_elem.cssselect('td')] for tr_elem in tr_elems[1:]]

        df = pd.DataFrame(raw_data, columns=['Date', 'CPI_Monthly'])
        # parse date formats
        df.Date = pd.to_datetime(df.Date, format='%b %d, %Y')
        # transform to numeric values
        df['CPI_Monthly'] = pd.to_numeric(df['CPI_Monthly'])
        # df.Price = pd.to_numeric(df.Price.str.replace(',', '').astype(float))  # handle commas inside strings
        df = df.set_index('Date')

        return df
