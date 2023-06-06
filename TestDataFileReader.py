from unittest import TestCase

from numpy import dtype

from DataFileReader import DataFileReader


class TestDataFileReader(TestCase):
    def test_read_us_market_visualizations(self):
        df = DataFileReader().read_us_market_visualizations()
        # verify columns 3 and higher are float
        for index, col in enumerate(df.columns):
            if index > 3:
                assert df[col].dtypes == dtype('float64')
