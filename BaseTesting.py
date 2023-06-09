import unittest

import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_basic_data_frame_and_dictionary(self):
        data = {'Key':['k1','k2','k3'],'Value':['A','B','C']}
        df1 = pd.DataFrame(data)
        df1.set_index('Key')
        print(df1)
        i = df1.index.get_loc('k2')
        print(i)

    def test_creation_of_dataframe(self):
        data = ['a','b','c']
        keys = ['k1','k2','k3']
        columns = ['c1','c2']
        df = pd.DataFrame(data=[keys,data],index=columns).T
        print(df)
