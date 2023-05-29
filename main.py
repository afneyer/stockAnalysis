# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open("USMarketVisualizations.csv") as f:
        print(f.readline())

df = pd.read_csv("USMarketVisualizations.csv",
                 sep=',',
                 header='infer',
                 usecols=[2, 13],
                 parse_dates=['FullDate'], index_col='FullDate',
                 # names=["Date", "SP500+Div"],
                 engine='python')

# df['SP+Div Monthly'].str.replace(' ', '')
df = df.astype({'SP+DivMonthly': float})


print(df.head())
print(df.dtypes)

# df.plot(subplots=True, figsize=(15,6))
#df.plot(y=["SP+Div Monthly"], figsize=(15,4))
plot = df.plot(y=['SP+DivMonthly'], figsize=(20,10), grid=True, logy=True)
plot.grid('on', which='minor', axis='y')
plt.show()