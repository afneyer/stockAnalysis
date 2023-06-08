from typing import Dict, List

import numpy as np
import pandas as pd
import quandl
import requests
from lxml import html as HTMLParser
from pandas import DataFrame

quandl.ApiConfig.api_key = 'iYippvXT8yzS7xK47yoh'


class MyData:
    # data types
    quandle = 'quandle'
    multpl = 'multpl'
    compute = 'compute'

    # data columns
    sp500_pe_ratio_month = 'SP500_PE_Ratio_Month'
    sp500_div_yield_month = 'SP500_Div_Yield_Month'
    sp500_real_price_month = 'SP500_Real_Price_Month'
    cpi_urban_month = 'CPI_Urban_Month'
    ten_year_treasury = 'Ten_Year_Treasury'
    sp500_div_reinvest_month = 'SP500_Div_Reinvest_Month'

    urls = [
        [sp500_pe_ratio_month, quandle, 'MULTPL/SP500_PE_RATIO_MONTH'],
        [sp500_div_yield_month, quandle, 'MULTPL/SP500_DIV_YIELD_MONTH'],
        [sp500_real_price_month, quandle, 'MULTPL/SP500_REAL_PRICE_MONTH'],
        [cpi_urban_month, multpl, 'https://www.multpl.com/cpi/table/by-month'],
        [ten_year_treasury, multpl, 'https://www.multpl.com/10-year-treasury-rate/table/by-month'],
        [sp500_div_reinvest_month, compute, '']
    ]



def adjust_dates_to_start_of_month(df):
    index_list = []
    for index, row in df.iterrows():
        if index < len(df) - 1:
            next_row = df.iloc[index + 1]
            date = row['Date']
            next_date = next_row['Date']
            if date.is_month_end:
                new_date = date + pd.Timedelta(days=1)
                if new_date == next_date:
                    index_list.append(index)
                else:
                    df.replace(date, new_date, inplace=True)
            elif not date.is_month_start:
                index_list.append(index)
        else:
            if not row['Date'].is_month_start:
                index_list.append(index)
    df.drop(index_list, inplace=True)
    return df


def change_to_row_index(df):
    # if date is the index change to row index
    if len(df) > 0:
        if type(df.index[0]) is pd.Timestamp:
            df['Date'] = df.index
            df['RowNum'] = np.arange(len(df))
            df.set_index('RowNum', inplace=True)


def restore_date_index(df):
    if len(df) >0:
        if df.index.name == 'RowNum':
            df.set_index('Date', inplace=True)


class DataImporter:

    data_dict: DataFrame
    # contains the list of loaded data frames accessed the name of the series
    data_sets: List

    # data columns
    def __init__(self):
        self.data_dict = pd.DataFrame(MyData.urls, columns=['series', 'type', 'url'])
        self.data_dict.set_index('series',inplace=True)
        self.data_sets = []

    def get_data_set(self,set_name):
        return self.data_sets[self.data_dict.index.get_loc(set_name)]
    def import_us_market(self) -> DataFrame:
        df: DataFrame
        df = quandl.get('MULTPL/SP500_PE_RATIO_MONTH', )
        x = df['Value']

        return df

    def get_data_from_quandle(self, url):
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
            df['Value'] = df['Value'].str.rstrip('%').astype('float') / 100.0
        else:
            df['Value'] = pd.to_numeric(df['Value'])
        df = df.set_index('Date')

        return df

    def compute_series(self, fs_name):
        if fs_name == MyData.sp500_div_reinvest_month:
            df_sp500 = self.get_data_set(MyData.sp500_real_price_month)
            df_div_yield = self.get_data_set(MyData.sp500_div_yield_month)
            return df_sp500

    def import_my_data(self):
        # data_frame_list = []
        for url in MyData.urls:
            fs_name = url[0]
            fs_type = url[1]
            url_str = url[2]
            if fs_type == 'quandle':
                df = self.get_data_from_quandle(url_str)
            elif fs_type == 'multpl':
                df = self.get_data_from_multpl_website(url_str)
            elif fs_type == 'compute':
                df = self.compute_series(fs_name)
            df = df.rename(columns={'Value': fs_name})
            change_to_row_index(df)
            df = adjust_dates_to_start_of_month(df)
            restore_date_index(df)
            # data_frame_list.append(df)
            self.data_sets.append(df)
            print(self.data_sets)
        # self.data_dict['dataFrame'] = data_frame_list

    '''
    # The following did not work, remove eventually TODO
    # for now it's sample code
    def correct_dates(df):
        raise Exception("Do not use this function, it does not work properly")

        # drop end of the month date if there is already a date for the first of the next month
        print(df.head)
        d1 = df['Date'].shift(1)
        print(d1.head)
        d2 = df['Date'] + pd.DateOffset(days=1)
        print(d2.head)
        df = df.drop(df.loc[df['Date'].shift(1).eq(df['Date'] + pd.DateOffset(days=-1))].index)
        print(df.head)
        # move end of the month dates to the first
        df['Date'] = df['Date'].apply(lambda x: x + pd.DateOffset(day=1))
        df.set_index('Date', inplace=True)
        return df
    '''
