import numpy as np
import pandas_datareader.data as web
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from pandas import DataFrame

from DataImporter import restore_date_index
from DfUtil import scale_to_start_value
from NumUtilities import total_return




def totalReturnGraph(tickers: list, period: str='max', interval='1d' ):
    df = pd.DataFrame()
    for ticker in tickers:
        all_data = yf.Ticker(ticker)
        hist = all_data.history(period='max')
        price = hist['Close']
        div = hist['Dividends']
        cap = hist['Capital Gains']
        total_div = (div + cap).squeeze().to_numpy()
        price = price.squeeze().to_numpy()
        tot_ret = total_return(price,total_div)
        columns = ['Date',ticker+' tot ret']
        data = np.array([hist['Close'].index,tot_ret]).T
        df_ticker = pd.DataFrame(data,columns=columns)
        df_ticker.set_index('Date', inplace=True)
        # print(df_ticker)
        if df.empty:
            df = df_ticker
        else:
            df = df.merge(df_ticker,how='inner',on='Date')
        restore_date_index(df)
    print(df)
    scale_to_start_value(df)
    print(df)
    df.plot.line(y=df.columns,figsize=(10,7.5),logy=True)
    plt.show()

