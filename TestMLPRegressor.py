from unittest import TestCase

import matplotlib
from numpy import int64
from sklearn.neural_network import MLPRegressor

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import DataPlotUtil
from DataImporter import DataImporter
from MyData import MyData

plt.style.use('seaborn-poster')


# %matplotlib inline
class TestMLPRegressor(TestCase):

    def test_regression_sample_from_python_numerical_methods(self):
        np.random.seed(0)
        x = 10 * np.random.rand(100)

        def model(x, sigma=0.3):
            fast_oscillation = np.sin(5 * x)
            slow_oscillation = np.sin(0.5 * x)
            noise = sigma * np.random.randn(len(x))

            return slow_oscillation + fast_oscillation + noise

        plt.figure(figsize=(10, 8))
        y = model(x)
        plt.errorbar(x, y, 0.3, fmt='o')

        mlp = MLPRegressor(hidden_layer_sizes=(200, 200, 200),
                           max_iter=2000, solver='lbfgs',
                           alpha=0.01, activation='tanh',
                           random_state=8)

        xfit = np.linspace(0, 10, 1000)
        ytrue = model(xfit, 0)
        yfit = mlp.fit(x[:, None], y).predict(xfit[:, None])

        plt.figure(figsize=(10, 8))
        plt.errorbar(x, y, 0.3, fmt='o')
        plt.plot(xfit, yfit, '-r', label='predicted',
                 zorder=10)
        plt.plot(xfit, ytrue, '-k', alpha=0.5, \
                 label='true model', zorder=10)
        plt.legend()
        plt.show()

    def test_regression_sample_sp500_monthly(self):
        required_series = [MyData.sp500_div_yield_month]
        di = DataImporter()
        df = di.get_selected_series_as_df(required_series)
        fig, ax = DataPlotUtil.plot_sp500_monthly_logscale(di, MyData.sp500_div_reinvest_month)

        # shorten the curve using only n periods
        n = 300

        x = df.index.values
        m = len(x) - n
        x = x[m:]
        y = df[MyData.sp500_div_yield_month].astype(float).to_numpy()
        y = y[m:]*1000

        mlp = MLPRegressor(hidden_layer_sizes=(2000, 2000, 2000), \
                           max_iter=2000, solver='lbfgs', \
                           alpha=0.01, activation='tanh', \
                           random_state=8)

        xint = x.astype(int64)
        n = len(xint)
        trained_model = mlp.fit(xint[:, None], y)
        yfit = trained_model.predict(xint[:, None])  # predict(x)
        print(yfit)

        ax.plot(x, y, 'c', label='target')
        ax.plot(x, yfit, 'b', label='fitted')

        fig.set_figheight(7.5)
        fig.set_figwidth(10)
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.show()
