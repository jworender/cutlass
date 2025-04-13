import os
import pandas as pd
import numpy as np
from pynmaps.DataObject import DataObject

def test_add_column():
    data = {'A': [1, 2, 3]}
    table = DataObject(data)
    table.add_column('B', [4, 5, 6])
    assert 'B' in table.data.columns

def test_summary_statistics():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    print("Checking to ensure that the summary statistics yield the correct values...")
    sumstat = table.summary_statistics()
    assert (float(sumstat.iloc[0,0]) == 3.0) & (float(sumstat.iloc[0,1]) == 3.0) & \
           (float(sumstat.iloc[1,0]) == 2.0) & (float(sumstat.iloc[1,1]) == 5.0) & \
           (float(sumstat.iloc[2,0]) == 1.0) & (float(sumstat.iloc[2,1]) == 1.0) & \
           (float(sumstat.iloc[3,0]) == 1.0) & (float(sumstat.iloc[3,1]) == 4.0) & \
           (float(sumstat.iloc[4,0]) == 1.5) & (float(sumstat.iloc[4,1]) == 4.5) & \
           (float(sumstat.iloc[5,0]) == 2.0) & (float(sumstat.iloc[5,1]) == 5.0) & \
           (float(sumstat.iloc[6,0]) == 2.5) & (float(sumstat.iloc[6,1]) == 5.5) & \
           (float(sumstat.iloc[7,0]) == 3.0) & (float(sumstat.iloc[7,1]) == 6.0) 

def test_save_csv():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    if os.path.exists("test.csv"):
        print("Removing test csv.")
        os.remove("test.csv")
    assert table.save_csv("test.csv")

def test_load_csv():
    table = DataObject()
    assert os.path.exists("test.csv")
    print("Testing csv load...")
    assert table.load_csv("test.csv")

def test_filter_data():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    ftable = table.filter_rows(table.data['A'] >= 2)
    assert float(ftable.data.iloc[0,0]) == 2.0

def test_drop_missing():
    data = {'A': [1, 2, pd.NA], 'B': [4, 5, 6]}
    table = DataObject(data)
    table.drop_missing()
    assert table.data.shape[0] == 2

def test_rescale_column():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    table.rescale_column('A')
    assert (float(np.min(table.data['A'])) == 0.0) & (float(table.data.iloc[1,0]) == 0.5) & \
           (float(np.max(table.data['A'])) == 1.0)
    table.rescale_column('A', min=1, max=5)
    assert (float(np.min(table.data['A'])) == 1.0) & (float(table.data.iloc[1,0]) == 3.0) & \
           (float(np.max(table.data['A'])) == 5.0)

def test_random_noise():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    d    = pd.DataFrame(data)
    table = DataObject(data)
    table.add_random_noise('A', std_dev = 0.1)
    diffs = d['A'] - table.data['A']
    assert all(diffs != 0)

def test_impute():
    data = {'A': [1, 2, pd.NA, 4, 5], 'B': [4, 5, 6, 7, 8], 'C': ['A', 'B', 'A', pd.NA, 'C']}
    table = DataObject(data)

data = {'A': [1, 2, 3, 4, 5, 6], 'B': [4, 5, 6, 6, 5, 4]}
table = DataObject(data)
table.plot('A', 'B')


