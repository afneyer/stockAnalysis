from typing import Dict, List

import numpy
import numpy as np
import pandas as pd
import quandl
import requests
from lxml import html as HTMLParser
from pandas import DataFrame

import NumUtilities

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
    sp500_earnings_growth = 'SP500_Growth_Based_On_Earnings'
    sp500_earnings_yield = "SP500_Earnings_Annual_Yield_Monthly"
    us_gdp_nominal = "Nominal_US_GDP_Quarterly"

    urls = [
        [sp500_pe_ratio_month, quandle, 'MULTPL/SP500_PE_RATIO_MONTH'],
        [sp500_div_yield_month, quandle, 'MULTPL/SP500_DIV_YIELD_MONTH'],
        [sp500_real_price_month, quandle, 'MULTPL/SP500_REAL_PRICE_MONTH'],
        [cpi_urban_month, multpl, 'https://www.multpl.com/cpi/table/by-month'],
        [ten_year_treasury, multpl, 'https://www.multpl.com/10-year-treasury-rate/table/by-month'],
        [us_gdp_nominal,quandle,'FRED/GDP'],
        [sp500_div_reinvest_month, compute, ''],
        [sp500_earnings_growth, compute,''],
        [sp500_earnings_yield, compute,'']
    ]

def adjust_dates_to_start_of_month(df):
    change_to_row_index(df)
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
    restore_date_index(df)
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


def adjust_sequence(id,df):
    if id == MyData.us_gdp_nominal:
        # add additional data points and use cubit spline interpolation to set data
        df1 = df.asfreq('MS')
        df2 = df1.interpolate(method='cubicspline')
        return df2
    elif id == MyData.sp500_div_yield_month:
        return adjust_dates_to_start_of_month()
    elif id == MyData.sp500_real_price_month:
        return adjust_dates_to_start_of_month()
    else:
        return df



class DataImporter:

    data_dict: DataFrame
    # contains the list of loaded data frames accessed the name of the series
    all_data: DataFrame

    # data columns
    def __init__(self):
        self.data_dict = pd.DataFrame(MyData.urls, columns=['series', 'type', 'url'])
        self.data_dict.set_index('series',inplace=True)
        self.all_data = pd.DataFrame()

    def get_numpy(self, set_name) -> numpy:
        return self.all_data[set_name].to_numpy()

    def get_index_values(self):
        return self.all_data.index.values;

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
            np_div_yield = self.get_numpy(MyData.sp500_div_yield_month)
            np_sp500 = self.get_numpy(MyData.sp500_real_price_month)
            np_div_value = np_div_yield * np_sp500 / 1200.0
            np_tot_return = NumUtilities.total_return(np_sp500,np_div_value)
            print(np_tot_return)
            print(self.get_index_values())
            df = pd.DataFrame(data=[self.get_index_values(),np_tot_return],
                              index=['Date',MyData.sp500_div_reinvest_month]).T
            df.set_index('Date',inplace=True)
            print(df)
            return(df)

        if fs_name == MyData.sp500_earnings_growth:
            np_pe = self.get_numpy(MyData.sp500_pe_ratio_month)
            np_sp500 = self.get_numpy(MyData.sp500_real_price_month)
            np_earnings_yield = np.reciprocal(np_pe) / 12.0
            np.savetxt("Earnings Yield.txt",np_earnings_yield)
            sp500_start = np_sp500[0]
            np_tot_earn_ret = NumUtilities.yield_return(sp500_start,np_earnings_yield)
            df = pd.DataFrame(data=[self.get_index_values(), np_tot_earn_ret],
                              index=['Date', MyData.sp500_earnings_growth]).T
            df.set_index('Date', inplace=True)
            return df

        if fs_name == MyData.sp500_earnings_yield:
            np_pe = self.get_numpy(MyData.sp500_pe_ratio_month)
            np_earnings_yield = np.reciprocal(np_pe)
            df = pd.DataFrame(data=[self.get_index_values(), np_earnings_yield],
                              index=['Date', MyData.sp500_earnings_yield]).T
            df.set_index('Date', inplace=True)
            return df

    def import_my_data(self):
        # data_frame_list = []
        for url in MyData.urls:
            fs_name = url[0]
            fs_type = url[1]
            url_str = url[2]
            print("---Importing: " + fs_name)
            if fs_type == 'quandle':
                df = self.get_data_from_quandle(url_str)
            elif fs_type == 'multpl':
                df = self.get_data_from_multpl_website(url_str)
            elif fs_type == 'compute':
                df = self.compute_series(fs_name)
            df = df.rename(columns={'Value': fs_name})

            # make adjustment for specific data series
            df = adjust_sequence(id,df)

            # add to the all_data frame
            if self.all_data.empty:
                self.all_data = df
            else:
                self.all_data = pd.merge(self.all_data,df,on='Date',how='outer')
            print(self.all_data.head)
            np.savetxt(fs_name+'.txt',df.values)
        return self.all_data

    def import_all_data(self,id_list):
        for id in id_list:
            url = MyData.urls

    def get_all_data(self):
        return self.all_data

    def get_url(self,id):
        return self.data_dict['url'].get(id)

    def get_url_type(self,id):
        return self.data_dict['type'].get(id)

