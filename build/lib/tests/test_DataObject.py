import pytest
import os
import pandas as pd
import numpy as np
from cutlass.DataObject import DataObject
from cutlass.lle import lle, sigmoid, accuracy_score

# Test 1
def test_add_column():
    data = {'A': [1, 2, 3]}
    table = DataObject(data)
    table.add_column('B', [4, 5, 6])
    assert 'B' in table.data.columns

# Test 2
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

# Test 3
def test_save_csv():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    if os.path.exists("test.csv"):
        print("Removing test csv.")
        os.remove("test.csv")
    assert table.save_csv("test.csv")

# Test 4
def test_load_csv():
    table = DataObject()
    assert os.path.exists("test.csv")
    print("Testing csv load...")
    assert table.load_csv("test.csv")

# Test 5
def test_filter_data():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    ftable = table.filter_rows(table.data['A'] >= 2)
    assert float(ftable.data.iloc[0,0]) == 2.0

# Test 6
def test_drop_missing():
    data = {'A': [1, 2, pd.NA], 'B': [4, 5, 6]}
    table = DataObject(data)
    table.drop_missing()
    assert table.data.shape[0] == 2

# Test 7
def test_rescale_column():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    table = DataObject(data)
    table.rescale_column('A')
    assert (float(np.min(table.data['A'])) == 0.0) & (float(table.data.iloc[1,0]) == 0.5) & \
           (float(np.max(table.data['A'])) == 1.0)
    table.rescale_column('A', min=1, max=5)
    assert (float(np.min(table.data['A'])) == 1.0) & (float(table.data.iloc[1,0]) == 3.0) & \
           (float(np.max(table.data['A'])) == 5.0)

# Test 8
def test_random_noise():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    d    = pd.DataFrame(data)
    table = DataObject(data)
    table.add_random_noise('A', std_dev = 0.1)
    diffs = d['A'] - table.data['A']
    assert all(diffs != 0)

# Test 9
def test_impute():
    data = {'A': [1, 2, pd.NA, 4, 5], 'B': [4, 5, 6, 7, 8], 'C': ['A', 'B', 'A', pd.NA, 'C']}
    table = DataObject(data)

# Test 10
def test_fit_model_linr():
    # Test fitting a model
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    })
    obj = DataObject(data=data)
    metrics = obj.fit_model(
        model_type='linear_regression',
        task='regression',
        X_columns=['feature1'],
        y_column='feature2',
        model_name='lin_reg_test'
    )
    assert 'mean_squared_error' in metrics, "Model metrics should include 'mean_squared_error'."

# Test 11
def test_fit_model_logr():
    # Test fitting a model
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    })
    obj = DataObject(data=data)
    metrics = obj.fit_model(
        model_type='logistic_regression',
        X_columns=['feature1', 'feature2'],
        y_column='target',
        model_name='log_reg_test'
    )
    assert 'accuracy' in metrics, "Model metrics should include 'accuracy'."
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1."

# Test 12
def test_predict_logr():
    data = pd.DataFrame({
        'example1':  [0.0, 0.51, 0.0, 0.10 ], # 0
        'example2':  [0.0, 0.85, 0.0, 0.50 ], # 1
        'example3':  [0.0, 0.85, 0.0, 0.00 ], # 0
        'example4':  [0.0, 0.80, 0.0, 0.65 ], # 1
        'example5':  [0.0, 0.55, 0.0, 0.59 ], # 1
        'example6':  [0.0, 0.15, 0.0, 0.52 ], # 0
        'example7':  [0.0, 0.05, 0.0, 0.50 ], # 0
        'example8':  [0.0, 0.85, 0.0, 0.55 ], # 1
        'example9':  [0.0, 0.85, 0.0, 0.00 ], # 0
        'example10': [0.0, 0.85, 0.0, 0.75 ], # 1
    }).T
    data.columns = ['feature1', 'feature2', 'feature3', 'feature4']
    data['target'] = [0, 1, 0, 1,  1, 0, 0, 1, 0, 1]
    
    obj = DataObject(data=data)
    obj.fit_model(
        model_type='logistic_regression',
        task='classification',
        X_columns=['feature1', 'feature2', 'feature3', 'feature4'],
        y_column='target',
        model_name='log_reg_test'
    )
    predictions = obj.predict(model_name='log_reg_test', X=obj.data[['feature1', 'feature2', 'feature3', 'feature4']])
    assert len(predictions) == len(data), "Prediction length does not match input data length."
    assert list(predictions) == list(data['target']), "Predictions elements do not match input data."

# Test 13
def test_predict_linr():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    })
    obj = DataObject(data=data)
    obj.fit_model(
        model_type='linear_regression',
        task='regression',
        X_columns=['feature1'],
        y_column='feature2',
        model_name='lin_reg_test'
    )
    predictions = obj.predict(model_name='lin_reg_test', X=obj.data[['feature1']])
    assert len(predictions) == len(data), "Prediction length does not match input data length."
    assert sum(np.array(predictions) - np.array(data['feature2'])) < 1E-6, "Prediction elements do not match input data."

# Test 14
def test_no_models():
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    obj = DataObject(data=data)
    predictions = obj.predict(X=obj.data[['feature1', 'feature2']])
    assert predictions is None, "Prediction should return None when no models are available."

# Test 15
def test_lle():

    # Hyperparameters for custom logistic regression with L1
    lambda_ = 0.01         # Regularization strength
    learning_rate = 0.02   # Learning rate
    max_iter = 1000        # Number of iterations
    batch_size = 10
    kfolds = 10
    verbose = False

    # define some sample data (the same data used in the paper)
    #                A   B   C   D   E   F   G   H   I   J
    X = np.array([[  1,  1,  1, -1,  1,  1,  1,  1,  1,  1 ],
                  [ -1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],
                  [ -1, -1, -1,  1, -1,  1,  1, -1, -1,  1 ],
                  [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],
                  [ -1,  1,  1,  1,  1,  1, -1,  1,  1,  1 ],
                  [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],
                  [  1, -1,  1, -1, -1, -1,  1,  1, -1,  1 ],
                  [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ],
                  [ -1, -1,  1,  1,  1,  1,  1,  1, -1,  1 ],
                  [ -1,  1, -1,  1,  1, -1, -1,  1, -1, -1 ]])

    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 0])

    data = pd.DataFrame(X)
    data.columns = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' ]
    data['target'] = y

    obj = DataObject(data)

    metrics = obj.fit_model(
        model_type='lle',
        X_columns=[ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' ],
        y_column='target',
        test_size=0,
        model_name='custom_lle',
        lambda_=0.01,
        learning_rate=0.02,
        max_iter=1000,
        early_exit=0.95,
        verbose=True
    )

    print(metrics)

    y_pred = obj.predict(model_name='custom_lle', X=X)
    print(y_pred)

    assert len(y_pred) == len(y), "Prediction length does not match input data length."
    assert list(y_pred) == list(y), "Predictions elements do not match input data."



