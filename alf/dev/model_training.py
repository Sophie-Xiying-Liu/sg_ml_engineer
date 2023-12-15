"""
This script trains the model.
"""

import os
import pathlib
import logging
import pandas as pd
import numpy as np
import yaml
import logging



# Set the current working directory to the directory of this file
if "__file__" not in locals() or __file__ == "<input>":
    __file__ = "/Users/Sophie/Documents/GitHub/sg_ml_engineer/alf/dev/model_training.py"

filedir = pathlib.Path(os.path.dirname(__file__))
datadir = filedir / "data"

# Load the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), "params.yaml")
try:
    # Open the YAML file and load its content
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    # Check if params is not None before subscripting
    if params is None:
        logging.error("Error: YAML file is empty or not loaded.")
except Exception as e:
    logging.error(f"Error loading YAML file: {e}")


def import_fe_file(fe_file, dt_col, datadir=datadir):
    """Import curated data"""
    data_file_path = os.path.join(datadir, fe_file)
    df = pd.read_csv(data_file_path)
    df[dt_col] = pd.to_datetime(df[dt_col])
    return df

def create_sets(df, dt_col, target_col, train_start, train_end):
    """Create training and test sets"""
    df_train = df[(df[dt_col] >= train_start) & (df[dt_col] <= train_end)]
    df_test = df[df[dt_col] > train_end]
    X_train = df_train.drop(columns=[dt_col, target_col])
    y_train = df_train[target_col]
    X_val = df_test.drop(columns=[dt_col, target_col])
    y_val = df_test[target_col]
    return X_train, y_train, X_val, y_val

