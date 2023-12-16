"""
Feature engineering scipt.
"""

import os
import pathlib
import logging
import pandas as pd
import numpy as np
import yaml
import holidays
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pickle import dump


# Set the current working directory to the directory of this file
if "__file__" not in locals() or __file__ == "<input>":
    __file__ = "/Users/Sophie/Documents/GitHub/sg_ml_engineer/alf/dev/fe_pipeline.py"

filedir = pathlib.Path(os.path.dirname(__file__))
datadir = filedir / "data"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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


def import_curated_data(curated_data_file, dt_col, datadir=datadir):
    """Import curated data"""
    data_file_path = os.path.join(datadir, curated_data_file)
    df = pd.read_csv(data_file_path)
    df[dt_col] = pd.to_datetime(df[dt_col])
    return df

def create_holiday(df, dt_col, country="CH"):
    """create holiday dataframe
    Args:
        country (str): country code
    Returns:
        pd: holiday dataframe
    """
    holidays_ch = holidays.CH(
        years=df[dt_col].dt.year.unique().tolist()
    )

    holiday_df = pd.DataFrame({
        "holiday_name": list(holidays_ch.values()),
        "holiday_date": list(holidays_ch.keys()),
    })

    return holiday_df

def create_time_features(df, dt_col, holiday_df):
    """create time features
    Args:
        df (pd): dataframe
        dt_col (str): datetime column
    Returns:
        pd: dataframe with time features
    """
    df = df.assign(
        hour=lambda x: x[dt_col].dt.hour + 1,
        month=lambda x: x[dt_col].dt.month,
        quarter=lambda x: x[dt_col].dt.quarter,
        wday=lambda x: x[dt_col].dt.day_of_week + 1,
        weekend=lambda x: np.where(
            x[dt_col].dt.day_name().isin(["Sunday", "Saturday"]), 1, 0
        ).astype(str),
        work_hour=lambda x: np.where(
            x["hour"].isin([19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7]), 0, 1
        ).astype(str),
        week_hour=lambda x: x[dt_col].dt.dayofweek * 24 + (x[dt_col].dt.hour + 1),
        year=lambda x: x[dt_col].dt.year,
    )\
        .assign(day=lambda x: x[dt_col].dt.date)\
            .merge(holiday_df, how="left", left_on="day", right_on="holiday_date")\
                .drop(["holiday_date", "day"], axis=1)\
                    .assign(
                        holiday_name = lambda x: np.where(
                            # x["holiday_name"].isna(), "none", x["holiday_name"]
                            x["holiday_name"].isna(), 0, 1
                            )
                            )
    return df

def cyclical_encoding(df):
    """Create cyclical encoding for time features."""
    def sin_transformer(period):
        """Create sin transformer."""
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(period):
        """Create cos transformer."""
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    # hour in day
    df["hour_sin"] = sin_transformer(24).fit_transform(df["hour"].astype(float))
    df["hour_cos"] = cos_transformer(24).fit_transform(df["hour"].astype(float))

    # hour in week
    df["week_hour_sin"] = sin_transformer(168).fit_transform(df["week_hour"].astype(float))
    df["week_hour_cos"] = cos_transformer(168).fit_transform(df["week_hour"].astype(float))

    # month
    df["month_sin"] = sin_transformer(12).fit_transform(df["month"].astype(float))
    df["month_cos"] = cos_transformer(12).fit_transform(df["month"].astype(float))

    # quarter
    df["quarter_sin"] = sin_transformer(4).fit_transform(df["quarter"].astype(float))
    df["quarter_cos"] = cos_transformer(4).fit_transform(df["quarter"].astype(float))

    # weekday
    df["wday_sin"] = sin_transformer(7).fit_transform(df["wday"].astype(float))
    df["wday_cos"] = cos_transformer(7).fit_transform(df["wday"].astype(float))

    df = df.drop(["hour", "month", "quarter", "wday", "week_hour"], axis=1)
    
    return df

def create_target_lags(df, dt_col, n_lag, target_col, fe_file):
    """Create target lags."""
    df[dt_col] = pd.to_datetime(df[dt_col])

    hours = df[dt_col].tolist()
    lag_lists = []

    for hour_idx, hour in enumerate(hours):
        prior_cutoff = hour - pd.Timedelta("1 hour")
        lags = df.query(f"{dt_col} <= @prior_cutoff").tail(n_lag)[target_col].tolist()
        lag_lists.append(lags)
    lag_df = pd.DataFrame(lag_lists).add_prefix("target_lag_")
    df = pd.concat([df, lag_df], axis=1)

    df = df.to_csv(os.path.join(datadir, fe_file), index=False)

    return df

def fe_pipeline(
        fe_file,
        dt_col,
        target_col,
        cat_cols,
        # label_cols,
        cyclical_cols,
        preprocessor_file,
        ):
    """Feature engineering pipeline."""
    df = pd.read_csv(os.path.join(datadir, fe_file))
    pipeline_cols = [
        col for col in df.columns if col not in [dt_col, target_col]
    ]
    num_cols = [
        col for col in pipeline_cols if col not in cat_cols\
            # and col not in label_cols\
            and col not in cyclical_cols
    ]
    

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    # label_transformer = Pipeline(
    #     steps=[
    #         ("encoder", LabelEncoder()),
    #     ]
    # )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, num_cols),
            ("categorical", categorical_transformer, cat_cols),
            # ("label", label_transformer, label_cols),
        ],
        remainder="passthrough",
    )

    logging.info("Preprocessor created.")

    dump(
        preprocessor,
        open(
            os.path.join(os.path.dirname(__file__), preprocessor_file), 'wb'
        )
    )

    return preprocessor


if __name__ == "__main__":
    CURATED_DATA_FILE = params["curated_data_file"]
    DT_COL = params["dt_col"]
    N_LAG = params["n_lag"]
    TARGET_COL = params["target_col"]
    CAT_COLS = params["cat_cols"]
    # LABEL_COLS = params["label_cols"]
    CYCLICAL_COLS = params["cyclical_cols"]
    FE_FILE = params["fe_file"]
    PREPROCESSOR_FILE = params["preprocessor_file"]

    df = import_curated_data(CURATED_DATA_FILE, dt_col=DT_COL)
    holiday_df = create_holiday(df, DT_COL)
    df = create_time_features(df, dt_col=DT_COL, holiday_df=holiday_df)\
        .pipe(cyclical_encoding)\
        .pipe(
            create_target_lags,
            dt_col=DT_COL,
            n_lag=N_LAG,
            target_col=TARGET_COL, 
            fe_file=FE_FILE
        )
    
    fe_pipeline(
        fe_file=FE_FILE,
        dt_col=DT_COL,
        target_col=TARGET_COL,
        cat_cols=CAT_COLS,
        cyclical_cols=CYCLICAL_COLS,
        # label_cols=LABEL_COLS,
        preprocessor_file=PREPROCESSOR_FILE
    )
