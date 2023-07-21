# Time Series Forecasting - Multivariate Time Series Models for Stock Market Prediction
# A tutorial for this file is available at www.relataly.com

import unittest
import math  # Mathematical functions
import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime  # Date Functions
from pandas.plotting import register_matplotlib_converters  # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt  # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates  # Formatting dates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Packages for measuring model performance / errors
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.preprocessing import RobustScaler, \
    MinMaxScaler  # This Scaler removes the median and scales the data according to the quantile range to normalize the price data
import seaborn as sns  # Visualization
from tensorflow.python.client import device_lib
import yfinance as yf  # Alternative package if webreader does not work: pip install yfinance
import random

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


# Setting the timeframe for the data extraction
# end_date =  date.today().strftime("%Y-%m-%d")
end_date = '2020-01-01'
start_date = '2010-01-01'

# Getting NASDAQ quotes
stockname = 'NASDAQ'
symbol = '^IXIC'


def import_data():
    # You can either use webreader or yfinance to load the data from yahoo finance
    # import pandas_datareader as webreader
    # df = webreader.DataReader(symbol, start=start_date, end=end_date, data_source="yahoo")
    df = yf.download(symbol, start=start_date, end=end_date)

    # Create a quick overview of the dataset
    print(df.head())
    return df


def explore_data(df):
    # Plot line charts
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / ncols, 0))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
    for i, ax in enumerate(fig.axes):
        sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    plt.show()


def preprocess_data(df):
    # Indexing Batches
    train_df = df.sort_values(by=['Date']).copy()

    # List of considered Features
    FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume'
                # , 'Month', 'Year', 'Adj Close'
                ]

    print('FEATURE LIST')
    print([f for f in FEATURES])

    # Create the dataset with features and filter the data to the list of FEATURES
    data = pd.DataFrame(train_df)
    data_filtered = data[FEATURES]

    # We add a prediction column and set dummy values to prepare the data for scaling
    # Note: the prediction is just the close, not shifted
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close']

    # Print the tail of the dataframe
    print(data_filtered_ext)

    # Get the number of rows in the data
    nrows = data_filtered.shape[0]

    # Convert the data to numpy values
    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))
    print(np_data.shape)

    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_filtered_ext['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)

    # Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = 50

    # Prediction Index
    index_Close = data.columns.get_loc("Close")

    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

    # Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]

    # The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, sequence_length time steps per sample, and 6 features
    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i - sequence_length:i, :])  # contains sequence_length values 0-sequence_length * columsn
            y.append(data[i, index_Close])  # contains the prediction values for validation,  for single-step prediction

        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y

    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

    # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # Validate that the prediction value and the input match up
    # The last close price of the second input sample should equal the first prediction value
    print(x_train[1][sequence_length - 1][index_Close])
    print(y_train[0])
    return x_train, y_train, x_test, y_test, scaler_pred, data_filtered_ext, train_data_len


def configure_model(x_train, y_train):
    # Configure the neural network model
    model = Sequential()

    # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
    n_neurons = x_train.shape[1] * x_train.shape[2]
    print(n_neurons, x_train.shape[1], x_train.shape[2])
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    # Training the model
    epochs = 1
    batch_size = 16
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test)
                        )

    # callbacks=[early_stop])
    return history


def plot_model_progress(history):
    # Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 5), sharex=True)
    sns.lineplot(data=history.history["loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    epochs = len(history.history['loss'])
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Train", "Test"], loc="upper left")
    plt.grid()
    plt.show()

def get_model_performance(model, scaler_pred, x_test, y_test):
    # Get the predicted values
    y_pred_scaled = model.predict(x_test)

    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

    # Mean Absolute Error (MAE)
    MAE = mean_absolute_error(y_test_unscaled, y_pred)
    print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
    print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

    return y_pred;

def plot_model_results(data_filtered_ext, train_data_len, y_pred):
    # The date from which on the date is displayed
    display_start_date = "2019-01-01"

    # Add the difference between the valid and predicted prices
    train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid.insert(1, "y_pred", y_pred, True)
    valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
    df_union = pd.concat([train, valid])

    # Zoom in to a closer timeframe
    df_union_zoom = df_union[df_union.index > display_start_date]

    # Create the lineplot
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("y_pred vs y_test")
    plt.ylabel(stockname, fontsize=18)
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)

    # Create the bar plot with the differences
    df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
    ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3,
            label='residuals', color=df_sub)
    plt.legend()
    plt.show()


def run_tutorial():
    df = import_data()
    explore_data(df)
    x_train, y_train, x_test, y_test, scaler, data_filtered_ext, train_data_len = preprocess_data(df)
    model = configure_model(x_train, y_train)
    history = train_model(model, x_train, y_train, x_test, y_test)
    plot_model_progress(history)
    y_pred = get_model_performance(model,scaler,x_test,y_test)
    plot_model_results(data_filtered_ext, train_data_len, y_pred)




def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']


class MyTestCase(unittest.TestCase):
    def test_devices(self):
        print('Tensorflow Version: ' + tf.__version__)
        print(tf.config.list_physical_devices())
        print(tf.config.list_logical_devices())
        print(get_available_devices())

    def test_run_tutorial(self):
        run_tutorial()

    def test_train_cpu(self):
        with tf.device('/cpu:0'):
            run_tutorial()

    def test_run_gpu(self):
        with tf.device('/gpu:0'):
            run_tutorial()


