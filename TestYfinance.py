from unittest import TestCase
import yfinance as yf


class MyTestCase(TestCase):
    def test_yfinance_spy(self):
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1d",auto_adjust=False,start="1993-01-29",end="2024-05-07")
        close = hist["Close"]
        print(spy)
        self.assertEqual(True, False)  # add assertion here


