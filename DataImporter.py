import fredapi as fa
import numpy
import numpy as np
import pandas as pd
import yfinance as yf
import quandl
import requests
from IPython.core.display_functions import display
from lxml import html as HTMLParser
from pandas import DataFrame, Series

import NumUtilities
from DfUtil import scale_to_start_value
from MyData import MyData

# API keys
frd = fa.Fred(api_key='b056b2f10d5f964dadde84be2f5e7b73')
quandl.ApiConfig.api_key = 'iYippvXT8yzS7xK47yoh'


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
    if len(df) > 0:
        if df.index.name == 'RowNum':
            df.set_index('Date', inplace=True)


def print_df(df: DataFrame, descript: str = 'No Description') -> None:
    print('----- DataFrame Info for ' + descript + '-----')
    df.info()
    print(df.columns)
    print(df.dtypes)
    print(df.index.values)
    df.describe()
    print(df)
    print('------ End DataFrame Info for ' + descript + '-----')


# noinspection PyMethodMayBeStatic
class DataImporter:
    __data_dict: DataFrame
    # contains the list of loaded data frames accessed by the name of the series
    __series_list: dict
    __all_data: DataFrame

    # data columns
    def __init__(self):
        self.__data_dict = pd.DataFrame(MyData.urls, columns=['fs_id', 'type', 'url'])
        self.__data_dict.set_index('fs_id', inplace=True)
        self.__all_data = pd.DataFrame()
        self.__series_list = {}

    def display_series_dictionary(self):
        display(self.__data_dict.to_string())

    def get_series_as_df(self, fs_id: str):
        if fs_id in self.__series_list:
            df = self.__series_list[fs_id]
        else:
            df = self.__import_series(fs_id)

        return df

    def get_series_as_series(self, fs_id: str) -> Series:
        return self.get_series_as_df(fs_id).squeeze()

    def get_series_as_numpy(self, fs_id) -> numpy:
        fs = self.get_series_as_series(fs_id)
        return fs.astype(float).to_numpy()

    def get_selected_series_as_df(self, id_list: list):
        df = DataFrame()
        for fs_id in id_list:
            if fs_id not in df.columns:
                df1 = self.get_series_as_df(fs_id)
                if df.empty:
                    df = df1
                    # print('---Setting initial df before merging')
                    # print_df(df)
                else:
                    # print('---Merging df with ' + fs_id)
                    # print_df(df,'Base Dataframe')
                    # print_df(df1,'Merging Dataframe')
                    df = pd.merge(df, df1, left_index=True, right_index=True, how='inner')
                    # print(df)

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
        # remove commas and empty strings before numeric conversion
        df = df.replace(r'^$', np.nan, regex=True)
        df['Value'] = df['Value'].str.replace(',', '')
        # check whether the series in percentage
        if pd.api.types.is_string_dtype(df['Value'].dropna()):
            if df['Value'].str.contains('%').any():
                df['Value'] = df['Value'].str.rstrip('%')
                df['Value'] = df['Value'].astype('float') / 100.0
            elif df['Value'].str.contains('trillion').any():
                df['Value'] = df['Value'].str.rstrip('trillion')
                df['Value'] = df['Value'].astype('float') * 1e12
            elif df['Value'].str.contains('million').any():
                df['Value'] = df['Value'].str.rstrip('million')
                df['Value'] = df['Value'].astype('float') * 1e6
            else:
                df['Value'] = pd.to_numeric(df['Value'])
        # reset the index
        df = df.set_index('Date')
        df = df.sort_index()

        return df

    def get_data_from_fred(self, url):
        series = frd.get_series(url)
        series.name = url
        df = series.to_frame(name='Value')
        df.index.name = 'Date'
        df = df.interpolate(method='cubicspline')
        return df

    def get_total_return_from_yahoo(self, url):
        'for yahoo the url is the ticker'
        df = pd.DataFrame()

        all_data = yf.Ticker(url)
        hist = all_data.history(period='max', interval='1d')
        price = hist['Close']
        div = hist['Dividends']
        cap = hist['Capital Gains']
        total_div = (div + cap).squeeze().to_numpy()
        price = price.squeeze().to_numpy()
        tot_ret = NumUtilities.total_return(price, total_div)
        columns = ['Date', 'Value']
        data = np.array([hist['Close'].index, tot_ret]).T
        df_ticker = pd.DataFrame(data, columns=columns)
        df_ticker.set_index('Date', inplace=True)
        # print(df_ticker)
        if df.empty:
            df = df_ticker
        else:
            df = df.merge(df_ticker, how='inner', on='Date')
        restore_date_index(df)
        scale_to_start_value(df)
        df.tz_localize(None)
        return df

    def compute_series(self, fs_id):
        # create a df of the dependant series to align the series
        if fs_id == MyData.sp500_div_reinvest_month:
            prereq = [MyData.sp500_div_yield_month, MyData.sp500_real_price_month]
            df = self.get_selected_series_as_df(prereq)
            np_div_yield = df[MyData.sp500_div_yield_month].astype(float).to_numpy()
            np_sp500 = df[MyData.sp500_real_price_month].astype(float).to_numpy()
            np_div_value = np_div_yield * np_sp500 / 12.0
            np_tot_return = NumUtilities.total_return(np_sp500, np_div_value)
            # print(np_tot_return)
            df = pd.DataFrame(data=[df.index.values, np_tot_return],
                              index=['Date', MyData.sp500_div_reinvest_month]).T
            df.set_index('Date', inplace=True)
            # print(df)
            return df

        if fs_id == MyData.sp500_earnings_growth:
            prereq = [MyData.sp500_pe_ratio_month, MyData.sp500_earnings_yield]
            df = self.get_selected_series_as_df(prereq)
            np_pe = self.get_series_as_numpy(MyData.sp500_pe_ratio_month)
            np_sp500 = self.get_series_as_numpy(MyData.sp500_real_price_month)
            np_earnings_yield = np.reciprocal(np_pe) / 12.0
            np.savetxt("Earnings Yield.txt", np_earnings_yield)
            sp500_start = np_sp500[0]
            np_tot_earn_ret = NumUtilities.yield_return(sp500_start, np_earnings_yield)
            df = pd.DataFrame(data=[df.index.values, np_tot_earn_ret],
                              index=['Date', MyData.sp500_earnings_growth]).T
            df.set_index('Date', inplace=True)
            return df

        if fs_id == MyData.sp500_earnings_yield:
            prereq = [MyData.sp500_pe_ratio_month]
            df = self.get_selected_series_as_df(prereq)
            np_pe = self.get_series_as_numpy(MyData.sp500_pe_ratio_month)
            np_earnings_yield = np.reciprocal(np_pe)
            df = pd.DataFrame(data=[df.index.values, np_earnings_yield],
                              index=['Date', MyData.sp500_earnings_yield]).T
            df.set_index('Date', inplace=True)
            df = df.astype('float')
            # print(df.dtypes)
            return df

    def __import_series(self, fs_id):

        # import the series based on the type/source of series
        fs_type = self.get_url_type(fs_id)
        url_str = self.get_url(fs_id)
        df = DataFrame()
        if fs_type == MyData.quandle:
            print("---Importing from Quandle: ", fs_id)
            df = self.get_data_from_quandle(url_str)
        elif fs_type == MyData.multpl:
            print("---Importing from Multpl: ", fs_id)
            df = self.get_data_from_multpl_website(url_str)
        elif fs_type == MyData.fred:
            print("---Importing from " + MyData.fred, fs_id)
            df = self.get_data_from_fred(url_str)
        elif fs_type == MyData.yahoo:
            print("---Importing from " + MyData.yahoo, fs_id)
            df = self.get_total_return_from_yahoo(url_str)
        elif fs_type == MyData.compute:
            print("---Computing Series: ", fs_id)
            df = self.compute_series(fs_id)
        df = df.rename(columns={'Value': fs_id})

        # interpolate all directly imported sequences, not computed ones
        if fs_type != MyData.compute:
            df = self.adjust_sequence(fs_id, df)

        # save the sequence in a text file
        df.to_csv('./output/' + fs_id + ".csv")
        # add to the all_data frame
        self.__series_list[fs_id] = df

        return df

    def adjust_sequence(self, fs_id, df):

        # adjustments based on general criteria
        # Todo questionable: assumes yahoo data is always daily
        if self.get_url_type(fs_id) == MyData.yahoo:
            df = df.astype({fs_id: float})
            df = df.asfreq('D')

        # adjustments based on specific series
        if fs_id == MyData.sp500_pe_ratio_month:
            df1 = adjust_dates_to_start_of_month(df)
        elif fs_id == MyData.us_gdp_nominal:
            # add additional data points and use cubit spline interpolation to set data
            df1 = df.asfreq('MS')
        elif fs_id == MyData.sp500_div_yield_month:
            df[MyData.sp500_div_yield_month] = df[MyData.sp500_div_yield_month] / 100.0
            df1 = adjust_dates_to_start_of_month(df)
        elif fs_id == MyData.sp500_real_price_month:
            df1 = adjust_dates_to_start_of_month(df)
        elif fs_id == MyData.ten_year_treasury_month:
            df1 = adjust_dates_to_start_of_month(df)
        else:
            df1 = df

        # always create missing values
        df2 = df1.interpolate(method='cubicspline')
        return df2

    def get_url(self, fs_id):
        return self.__data_dict['url'].get(fs_id)

    def get_url_type(self, fs_id):
        return self.__data_dict['type'].get(fs_id)


def all_dates_month_start(df):
    return False not in (pd.to_datetime(df.index.values).day == 1)


# checks whether the dates in the index are all consecutive
def check_all_dates_daily(df) -> bool:
    dates = pd.to_datetime(df.index.values)
    dates_0: numpy = dates[1:]
    dates_1: numpy = dates + pd.Timedelta(1, "d")
    dates_1 = dates_1[:-1]
    all_daily = (dates_0 == dates_1)
    return False not in all_daily


def check_all_dates_daily_contiguous(df: DataFrame) -> bool:
    df1 = df[~df[df.columns[0]].isna()]
    return check_all_dates_daily(df1)


