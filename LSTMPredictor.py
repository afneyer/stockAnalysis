import os
import pandas as pd
import random
import tensorflow as tf
import time
import yahoo_fin.stock_info as si
from collections import deque
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
# import tensorflow.python.keras.models
# from keras.layers import Bidirectional
from matplotlib import pyplot as plt
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import LSTM, Dense, Dropout
# from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import numpy as np


class LSTMPredictor:
    def __init__(self):
        plt.style.use('seaborn-poster')

        # set seed, so we can get the same results after rerunning several times
        np.random.seed(314)
        tf.random.set_seed(314)
        random.seed(314)

        # Window size or the sequence length
        self.N_STEPS = 50
        # Lookup step, 1 is the next day
        self.LOOKUP_STEP = 15
        # whether to scale feature columns & output price as well
        self.SCALE = True
        self.scale_str = f"sc-{int(self.SCALE)}"
        # whether to shuffle the dataset
        self.SHUFFLE = True
        self.shuffle_str = f"sh-{int(self.SHUFFLE)}"
        # whether to split the training/testing set by date
        self.SPLIT_BY_DATE = True  # originally False
        self.split_by_date_str = f"sbd-{int(self.SPLIT_BY_DATE)}"
        # test ratio size, 0.2 is 20%
        self.TEST_SIZE = 0.2
        # features to use
        self.FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
        # date now
        self.date_now = time.strftime("%Y-%m-%d")
        self.date_now = "2020-01-01"  # todo review
        # model parameters
        self.N_LAYERS = 2
        # LSTM cell
        self.CELL = LSTM
        # 256 LSTM neurons
        self.UNITS = 256
        # 40% dropout
        self.DROPOUT = 0.4
        # whether to use bidirectional RNNs
        self.BIDIRECTIONAL = False
        # training parameters
        # mean absolute error loss
        # LOSS = "mae"
        # huber loss
        self.LOSS = "huber_loss"
        self.OPTIMIZER = "adam"
        self.BATCH_SIZE = 64
        self.EPOCHS = 2
        # EPOCHS = 500
        # Amazon stock market
        self.ticker = "AMZN"
        self.ticker_data_filename = os.path.join("data", f"{self.ticker}_{self.date_now}.csv")
        # model name to save, making it as unique as possible based on parameters
        self.model_name = f"{self.date_now}_{self.ticker}-{self.shuffle_str}-{self.scale_str}-{self.split_by_date_str}-\
        {self.LOSS}-{self.OPTIMIZER}-{self.CELL.__name__}-seq-{self.N_STEPS}-step-{self.LOOKUP_STEP}-layers-{self.N_LAYERS}-units-{self.UNITS}"
        if self.BIDIRECTIONAL:
            self.model_name += "-b"

    def shuffle_in_unison(self, a, b):
        # shuffle two arrays in the same way
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)

    def create_dirs(self):
        # create these folders if they does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir("data"):
            os.mkdir("data")

    def load_data(self, ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                  test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model,
            default is everything grabbed from yahoo_fin
        """
        # see if ticker is already a loaded stock from yahoo finance
        if isinstance(ticker, str):
            # load it from yahoo_fin library
            df = si.get_data(ticker)
        elif isinstance(ticker, pd.DataFrame):
            # already loaded, use it directly
            df = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        # this will contain all the elements we want to return from this function
        result = {'df': df.copy()}
        # we will also return the original dataframe itself
        # make sure that the passed feature_columns exist in the dataframe
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."
        # add date as a column
        if "date" not in df.columns:
            df["date"] = df.index
        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler
            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['adjclose'].shift(-lookup_step)
        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))
        # drop NaNs
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=n_steps)
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])
        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        # add to result
        result['last_sequence'] = last_sequence
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        if split_by_date:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - test_size) * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"] = X[train_samples:]
            result["y_test"] = y[train_samples:]
            if shuffle:
                # shuffle the datasets for training (if shuffle parameter is set)
                self.shuffle_in_unison(result["X_train"], result["y_train"])
                self.shuffle_in_unison(result["X_test"], result["y_test"])
        else:
            # split the dataset randomly
            result["X_train"], result["X_test"], result["y_train"], \
                result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
        return result

    def create_model(self, sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                     loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True),
                                            batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model

    def plot_graph(self, test_df):
        """
        This function plots true close price along with predicted close price
        with blue and red colors respectively
        """
        plt.plot(test_df[f'true_adjclose_{self.LOOKUP_STEP}'], c='b')
        plt.plot(test_df[f'adjclose_{self.LOOKUP_STEP}'], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()

    def get_final_df(self, model, data):
        """
        This function takes the `model` and `data` dict to
        construct a final dataframe that includes the features along
        with true and predicted prices of the testing dataset
        """
        # if predicted future price is higher than the current,
        # then calculate the true future price minus the current price, to get the buy profit
        buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
        # if the predicted future price is lower than the current price,
        # then subtract the true future price from the current price
        sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
        X_test = data["X_test"]
        y_test = data["y_test"]
        # perform prediction and get prices
        y_pred = model.predict(X_test)
        if self.SCALE:
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        test_df = data["test_df"]
        # add predicted future prices to the dataframe
        test_df[f"adjclose_{self.LOOKUP_STEP}"] = y_pred
        # add true future prices to the dataframe
        test_df[f"true_adjclose_{self.LOOKUP_STEP}"] = y_test
        # sort the dataframe by date
        test_df.sort_index(inplace=True)
        final_df = test_df
        # add the buy profit column
        final_df["buy_profit"] = list(map(buy_profit,
                                          final_df["adjclose"],
                                          final_df[f"adjclose_{self.LOOKUP_STEP}"],
                                          final_df[f"true_adjclose_{self.LOOKUP_STEP}"])
                                      # since we don't have profit for last sequence, add 0's
                                      )
        # add the sell profit column
        final_df["sell_profit"] = list(map(sell_profit,
                                           final_df["adjclose"],
                                           final_df[f"adjclose_{self.LOOKUP_STEP}"],
                                           final_df[f"true_adjclose_{self.LOOKUP_STEP}"])
                                       # since we don't have profit for last sequence, add 0's
                                       )
        return final_df

    def predict(self, model, data):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-self.N_STEPS:]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if self.SCALE:
            predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        return predicted_price

    def run_model_fit(self):
        # create the necessary directories
        self.create_dirs()
        # load the data
        data = self.load_data(self.ticker, self.N_STEPS, scale=self.SCALE, split_by_date=self.SPLIT_BY_DATE,
                              shuffle=self.SHUFFLE, lookup_step=self.LOOKUP_STEP, test_size=self.TEST_SIZE,
                              feature_columns=self.FEATURE_COLUMNS)
        # save the dataframe
        data["df"].to_csv(self.ticker_data_filename)
        # construct the model
        model = self.create_model(self.N_STEPS, len(self.FEATURE_COLUMNS), loss=self.LOSS, units=self.UNITS,
                                  cell=self.CELL, n_layers=self.N_LAYERS,
                                  dropout=self.DROPOUT, optimizer=self.OPTIMIZER, bidirectional=self.BIDIRECTIONAL)
        # some tensorflow callbacks
        checkpointer = ModelCheckpoint(os.path.join("results", self.model_name + ".h5"), save_weights_only=True,
                                       save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))
        # train the model and save the weights whenever we see
        # a new optimal model using ModelCheckpoint
        history = model.fit(data["X_train"], data["y_train"],
                            batch_size=self.BATCH_SIZE,
                            epochs=self.EPOCHS,
                            validation_data=(data["X_test"], data["y_test"]),
                            callbacks=[checkpointer, tensorboard],
                            verbose=1)
        return data, model

    def run_model_evaluation(self, data, model):
        # load optimal model weights from results folder
        model_path = os.path.join("results", self.model_name) + ".h5"
        model.load_weights(model_path)

        # evaluate the model
        loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        # calculate the mean absolute error (inverse scaling)
        if self.SCALE:
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
        else:
            mean_absolute_error = mae

        # get the final dataframe for the testing set
        final_df = self.get_final_df(model, data)

        # predict the future price
        future_price = self.predict(model, data)

        # we calculate the accuracy by counting the number of positive profits
        accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(
            final_df)
        # calculating total buy & sell profit
        total_buy_profit = final_df["buy_profit"].sum()
        total_sell_profit = final_df["sell_profit"].sum()
        # total profit by adding sell & buy together
        total_profit = total_buy_profit + total_sell_profit
        # dividing total profit by number of testing samples (number of trades)
        profit_per_trade = total_profit / len(final_df)

        # printing metrics
        print(f"Future price after {self.LOOKUP_STEP} days is {future_price:.2f}$")
        print(f"{self.LOSS} loss:", loss)
        print("Mean Absolute Error:", mean_absolute_error)
        print("Accuracy score:", accuracy_score)
        print("Total buy profit:", total_buy_profit)
        print("Total sell profit:", total_sell_profit)
        print("Total profit:", total_profit)
        print("Profit per trade:", profit_per_trade)

        # plot true/pred prices graph
        self.plot_graph(final_df)

        print(final_df.tail(10))
        # save the final dataframe to csv-results folder
        csv_results_folder = "csv-results"
        if not os.path.isdir(csv_results_folder):
            os.mkdir(csv_results_folder)
        csv_filename = os.path.join(csv_results_folder, self.model_name + ".csv")
        final_df.to_csv(csv_filename)

    def get_available_devices(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']
