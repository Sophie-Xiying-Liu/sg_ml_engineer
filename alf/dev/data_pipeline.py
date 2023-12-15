"""
This script cleans and preprocesses the input data.
"""

import os
import pathlib
import logging
import pandas as pd
import numpy as np
import yaml



# Set the current working directory to the directory of this file
if "__file__" not in locals() or __file__ == "<input>":
    __file__ = "/Users/Sophie/Documents/GitHub/sg_ml_engineer/alf/dev/data_pipeline.py"

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

# Set up logging
logging.basicConfig(filename='log_file', level=logging.ERROR)


def create_df(data_files, datadir=datadir):
    """ Creates the dataset.
    Args:
        data_files (dict): The data files directory.
        datadir (str): The data directory.
    Returns:
        pd: The dataset.
    """

    # Import active losses data
    active_losses = pd.read_csv(os.path.join(datadir, data_files["active_losses"]))
    active_losses.columns = active_losses.iloc[0, ]
    active_losses = active_losses.drop(active_losses.index[0])\
            .assign(
                Zeitstempel=lambda x: (
                    pd.to_datetime(x["Zeitstempel"]) - pd.Timedelta("15 minutes")
            ).dt.floor(freq="H"),
            kWh=lambda x: pd.to_numeric(x["kWh"]) / 1000,
        )\
            .groupby("Zeitstempel")\
                .agg(MWh=("kWh", "sum"))\
                    .reset_index()
    active_losses = active_losses.rename(
        columns={"Zeitstempel": "datetime", "MWh": "active_losses"}
    )

    # Import ntc data
    ntc = pd.read_csv(os.path.join(datadir, data_files["ntc"]), parse_dates=["datetime"])

    # # Import renewable data
    renewables = pd.read_csv(os.path.join(datadir, data_files["renewables"]), parse_dates=["datetime"])
    postfix = " [MW]"
    renewables.columns = renewables.columns.str.replace(f"{postfix}$", "")

    # Didn't import temperature data due to data missing
    # temperature = pd.read_csv(os.path.join(datadir, data_files["temperature"]), parse_dates=["datetime"])
    # temperature = temperature.assign(datetime=lambda x: x["datetime"] - pd.Timedelta("1 hour"))

    # Merge dataframes
    df = active_losses.merge(ntc, on='datetime', how='left')\
        .merge(renewables, on='datetime', how='left')\
        # .merge(temperature, on='datetime', how='left')
    
    return df

def check_timestamp(df, col="datetime"):
    """ Checks if the timestamp is complete and without duplicates.
    Args:
        df (pd): The dataset.
        col (str): The column name of the timestamp.
    Returns:
        bool: True if the timestamp is continuous, False otherwise.
    """
    # get the first and last timestamp
    print("check_timestamp")

    first_ts = df[col].min()
    last_ts = df[col].max()
    complete_timestamp = pd.date_range(first_ts, last_ts, freq="H")

    # Check duplicates
    df_duplicates = df[df[col].duplicated()]
    if not df_duplicates.empty:
        logging.error(
            f"Found duplicate timestamps: {df_duplicates[col].unique()}"
        )
        # Drop duplicates
        df = df[~df[col].duplicated()]

    # # Check if the timestamp is complete
    df_missing = complete_timestamp.difference(df[col])
    if not df_missing.empty:
        logging.error(
            f"Found missing timestamps: {df_missing}"
        )
        # Fill missing timestamps 
        df_missing = pd.DataFrame(
            index=df_missing, columns=df.columns, data=np.nan
        )
        df = df.append(df_missing)
        df = df.interpolate(method="polynomial", order=2, limit_direction="both")
        print("interpolate")

    return df

def clean_data(df):
    """ Cleans the dataset.
    Args:
        df (pd): The dataset.
    Returns:
        pd: The cleaned dataset.
    """
    # Check and fill missing values
    print("clean_data")
    nan_locations = df.iloc[:, 1:].isna().any(axis=1)
    print(df[nan_locations])
    # for column in nan_locations.columns:
    #     for index, is_nan in nan_locations[column].iteritems():
    #         if is_nan:
    #             logging.info(f"NaN value found in column '{column}' at index {index}")

    # Interpolate missing values
    # df = df.interpolate(method="polynomial", order=2, limit_direction="both")

    return df

if __name__ == "__main__":
    DATA_FILES = params["data_files"]
    create_df(data_files=DATA_FILES)\
        .pipe(check_timestamp)\
            .pipe(clean_data)
