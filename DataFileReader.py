import pandas as pd
from pandas import DataFrame


class DataFileReader:

    def read_us_market_visualizations(self) -> DataFrame:
        datafile = "./inputdata/USMarketVisualizations.csv"
        with open(datafile) as f:
            print(f.readline())

        df = pd.read_csv(datafile,
                         sep=',',
                         header='infer',
                         parse_dates=['FullDate'],
                         engine='python')
        df.reset_index(inplace=True)

        for index, col in enumerate(df.columns):
            if index >3:
                df[col] = df[col].astype(float)

        # Earnings come in a yearly earnings
        df['Earnings'] = df['Earnings'].div(12.0)
        df['SP500+DivMonthly'] = df['SP500+DivMonthly'] * df['SP500Index'].iloc[0]
        return df
