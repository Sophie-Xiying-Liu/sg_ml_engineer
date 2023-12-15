"""
This script cleans and preprocesses the input data.
"""

import os
import pathlib
import pandas as pd
import yaml


if "__file__" not in locals() or __file__ == "<input>":
    __file__ = "/Users/Sophie/Documents/GitHub/sg_ml_engineer/alf/dev/data_pipeline.py"

filedir = pathlib.Path(os.path.dirname(__file__))
datadir = filedir / "data"

yaml_path = os.path.join(os.path.dirname(__file__), "params.yaml")
try:
    # Open the YAML file and load its content
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)
        print(params)

    # Check if params is not None before subscripting
    if params is None:
        print("Error: YAML file is empty or not loaded.")
except Exception as e:
    print(f"Error loading YAML file: {e}")



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


if __name__ == "__main__":
    DATA_FILES = params["data_files"]

    create_df(data_files=DATA_FILES)