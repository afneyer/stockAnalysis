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
    ten_year_treasury = 'Ten_Year_Treasury'

    urls = [
        [sp500_pe_ratio_month, quandle, 'MULTPL/SP500_PE_RATIO_MONTH'],
        [sp500_div_yield_month, quandle, 'MULTPL/SP500_DIV_YIELD_MONTH'],
        [sp500_real_price_month, quandle, 'MULTPL/SP500_REAL_PRICE_MONTH'],
        [cpi_urban_month, multpl, 'https://www.multpl.com/cpi/table/by-month'],
        [ten_year_treasury,multpl, 'https://www.multpl.com/10-year-treasury-rate/table/by-month']
    ]


def correct_dates(df):
    # drop end of the month date if there is already a date for the first of the next month
    df = df.drop(df.loc[df['Date'].shift(1).eq(df['Date']+pd.DateOffset(days=-1))].index)
    # move end of the month dates to the first
    df['Date'] = df['Date'].apply(lambda x: x+pd.DateOffset(day=1))

    return df

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

        df = pd.DataFrame(raw_data, columns=['Date', 'Value'])
        # parse date formats
        df.Date = pd.to_datetime(df.Date, format='%b %d, %Y')

        # remove commas inside the values
        df['Value'] = df['Value'].str.rstrip(',')

        # transform to numeric values
        # check whether the series in percentage
        if df['Value'].str.contains('%').any():
            df['Value'] = df['Value'].str.rstrip('%').astype('float')/100.0
        else:
            df['Value'] = pd.to_numeric(df['Value'])
        df = df.set_index('Date')

        return df

    def import_my_data(self):
        for url in MyData.urls:
            fi_name = url[0]
            fi_type = url[1]
            url_str = url[2]
            if fi_type == 'quandle':
                df = self.get_data_from_quandle(url_str)
            elif fi_type == 'multpl':
                 df = self.get_data_from_multpl_website(url_str)
            df = df.rename(columns={'Value': fi_name})
            df = apply_specific_data_fixes(df)
            print(df.dtypes)
            print(df.head)



