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
import pprint


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

def create_multi_step_output(df, dt_col, target_col, n_forecast, input_file):
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

    df.to_csv(os.path.join(datadir, input_file), index=False)

    return df

def create_sets(df, dt_col, target_col, train_start, train_end):
    """Create training and test sets"""
    # Clean up the data
    df.dropna(inplace=True) 
    df.sort_values(dt_col, inplace=True)
    df = df.reset_index(drop=True) 

    # Split into train and test
    df_train = df[(df[dt_col] >= train_start) & (df[dt_col] <= train_end)]
    df_test = df[df[dt_col] > train_end]

    # Create X and y
    pipeline_cols = [
        col for col in df.columns if col not in [dt_col, target_col]
    ]

    target_cols = [col for col in pipeline_cols if col.startswith("target_forward_")]
    feature_cols = [col for col in pipeline_cols if col not in target_cols]

    X_train = df_train[feature_cols]
    y_train = df_train[target_cols]

    X_test = df_test[feature_cols]
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
    FE_FILE = params["fe_file"]
    DT_COL = params["dt_col"]
    TARGET_COL = params["target_col"]
    N_FORECAST = params["n_forecast"]
    INPUT_FILE = params["input_file"]
    TRAIN_START = params["train_start"]
    TRAIN_END = params["train_end"]
    PREPROCESSOR_FILE = params["preprocessor_file"]


    df = import_fe_file(
        fe_file=FE_FILE,
        dt_col=DT_COL,
        datadir=datadir,
    )\
    .pipe(create_multi_step_output,
        dt_col=DT_COL,
        target_col=TARGET_COL,
        n_forecast=N_FORECAST,
        input_file=INPUT_FILE
    )

    # X_train, y_train, X_test, y_test = create_sets(
    #     df,
    #     dt_col=DT_COL,
    #     target_col=TARGET_COL,
    #     train_start=TRAIN_START,
    #     train_end=TRAIN_END,
    # )
    
    # X_train_transformed, X_test_transformed = fe_transform(
    #     X_train,
    #     preprocessor_file=PREPROCESSOR_FILE,
    # )

    print(df.head())
    print(df.tail())
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.columns.to_list())