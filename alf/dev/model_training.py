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
from pickle import load


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

def create_multi_step_output(df, dt_col, target_col, n_forecast):
    """Create multi outputs."""
    df[dt_col] = pd.to_datetime(df[dt_col])

    hours = df[dt_col].tolist()
    lag_lists = []
    for hour_idx, hour in enumerate(hours):
        prior_cutoff = hour
        lags = df.query(f"{dt_col} >= @prior_cutoff").head(n_forecast)[target_col].tolist()
        lag_lists.append(lags)
    lag_df = pd.DataFrame(lag_lists).add_prefix("target_forward_")
    df = pd.concat([df, lag_df], axis=1)

    return df
   

def create_sets(df, dt_col, target_col, train_start, train_end):
    """Create training and test sets"""
    df.dropna(inplace=True) # drop rows with NaNs caused by lagging
    df.sort_values(dt_col, inplace=True)
    # Split into train and test
    df_train = df[(df[dt_col] >= train_start) & (df[dt_col] <= train_end)]
    df_test = df[df[dt_col] > train_end]

    target_cols = [col for col in df_train.columns if col.startswith("target_forward_")]
    X_train = df_train.drop(columns=[dt_col, target_col, target_cols])
    y_train = df_train[target_cols]

    X_test = df_test.drop(columns=[dt_col, target_col, target_cols])
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test

def fe_transform(X_train, preprocessor_file):
    """FE fit transform"""
    # Fit preprocess on X_train
    preprocessor_file_path = os.path.join(os.path.dirname(__file__), preprocessor_file)
    preprocessor = load(open(preprocessor_file_path, 'rb'))
    fitted_preprocessor = preprocessor.fit(X_train)

    X_train = pd.DataFrame(
            fitted_preprocessor.transform(X_train),
            columns=fitted_preprocessor.get_feature_names_out(),
        )

    X_test = pd.DataFrame(
        fitted_preprocessor.transform(X_test),
        columns=fitted_preprocessor.get_feature_names_out(),
    )

    return X_train, X_test

def train_model(X_train, y_train, X_val, y_val, model, model_name):
    """Train model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred


if __name__ == "__main__":
