# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os

# from utils import load_data, rename_columns, handle_missing_values, encode_target_variable, feature_engineering, preprocess_data_ml
from .utils import load_data, rename_columns, handle_missing_values, encode_target_variable, feature_engineering, preprocess_data_ml


def preprocess():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

    file_path = 'Churn_Data_-_Final_Version.csv'
    file_path = os.path.join(BASE_DIR, 'data', file_path)

    data = load_data(file_path)
    data = rename_columns(data)
    data = handle_missing_values(data, ['totalcharges_2017', 'totalcharges_2018', 'totalcharges_2019', 'totalcharges_2020', 'totalcharges_2023'])
    data = encode_target_variable(data)
    data = feature_engineering(data)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_data_ml(data)
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, data
    
def preprocessing_ml(data):
    print(data)
    return data