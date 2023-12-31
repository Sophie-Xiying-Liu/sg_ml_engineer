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
from pickle import load, dump
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import RegressorChain
import optuna
from optuna.samplers import TPESampler


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
    y_test = df_test[target_cols]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def fe_transform(X_train, X_test, preprocessor_file):
    """FE fit transform"""
    # Fit preprocess on X_train
    preprocessor_file_path = os.path.join(os.path.dirname(__file__), preprocessor_file)
    preprocessor = load(open(preprocessor_file_path, 'rb'))
    fitted_preprocessor = preprocessor.fit(X_train)

    X_train_transformed = pd.DataFrame(
            fitted_preprocessor.transform(X_train),
            columns=fitted_preprocessor.get_feature_names_out(),
        )
    print(f"X_train_transformed shape: {X_train_transformed.shape}")

    X_test_transformed = pd.DataFrame(
        fitted_preprocessor.transform(X_test),
        columns=fitted_preprocessor.get_feature_names_out(),
    )
    print(f"X_test_transformed shape: {X_test_transformed.shape}")

    return X_train_transformed, X_test_transformed

def train_model(X_train_transformed, y_train):
    """Train model"""

    # define the lgbm objective fct for Optuna tuning
    def objective(trial):
        lgbm_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 7000, 9000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.07),
            "num_leaves": trial.suggest_int("num_leaves", 44, 50),
            # "min_child_samples": trial.suggest_int("min_child_samples", 25, 30),
            # "min_child_weight": trial.suggest_float("min_child_weight", 0.00045, 0.00055),
            # "subsample": trial.suggest_float("subsample", 0.60, 0.70),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.9, 1.0),
        }

        metrics_dict = {
            "rmse": [],
            "mae": [],
            "r2": [],
            "mape": [],
        }

        tss = TimeSeriesSplit(n_splits=3)
        for train_index, val_index in tss.split(X_train_transformed):
            X_train_cv, X_val_cv = X_train_transformed.iloc[train_index], X_train_transformed.iloc[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

            print(X_train_cv.shape, X_val_cv.shape, y_train_cv.shape, y_val_cv.shape)

            lgbm_reg = lgbm.LGBMRegressor(**lgbm_params)
            model = RegressorChain(
                base_estimator=lgbm_reg,
                order=[i for i in range(0, 24)],
            )
            model.fit(X_train_cv, y_train_cv)

            y_val_pred = model.predict(X_val_cv)

            rmse = np.sqrt(mean_squared_error(y_val_cv.values.flatten(), y_val_pred.flatten()))
            mae = mean_absolute_error(y_val_cv.values.flatten(), y_val_pred.flatten())
            r2 = r2_score(y_val_cv.values.flatten(), y_val_pred.flatten())
            mape = mean_absolute_percentage_error(y_val_cv.values.flatten(), y_val_pred.flatten())

            metrics_dict['rmse'].append(rmse)
            metrics_dict['mae'].append(mae)
            metrics_dict['r2'].append(r2)
            metrics_dict['mape'].append(mape)
        
        average_performance = {
            metric: np.mean(metrics_dict[metric]) for metric in metrics_dict
        }
        print(f"Average Performance: {average_performance}")
            
    # Define Optuna's lgbm tuner
    sampler = TPESampler(seed=42) # use TPESampler for Bayesian optimization
    study = optuna.create_study(sampler=sampler, direction='minimize', study_name="losses_forecast_lgbm")
    study.optimize(
        objective,
        n_trials=10,
        show_progress_bar=True,
    )

    # get the best params
    best_params = study.best_params

    return best_params

def final_train(
        best_params,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name,
        metrics_file,
    ):
    """Train final model"""
    model = RegressorChain(
            base_estimator=lgbm.LGBMRegressor(**best_params),
            order=[i for i in range(0, 24)], # n_output=24
            random_state=42,
        )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test.values.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_test.values.flatten(), y_pred.flatten())
    r2 = r2_score(y_test.values.flatten(), y_pred.flatten())
    mape = mean_absolute_percentage_error(y_test.values.flatten(), y_pred.flatten())

    metrics_dict = {
            "rmse": [],
            "mae": [],
            "r2": [],
            "mape": [],
        }
    
    metrics_dict['rmse'].append(rmse)
    metrics_dict['mae'].append(mae)
    metrics_dict['r2'].append(r2)
    metrics_dict['mape'].append(mape)

    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(os.path.join(os.path.dirname(__file__), metrics_file), index=False)

    model_path = os.path.join(os.path.dirname(__file__), f"models/{model_name}.pickle")
    with open(model_path, 'wb') as file:
        dump(model, file)
    


if __name__ == "__main__":
    FE_FILE = params["fe_file"]
    DT_COL = params["dt_col"]
    TARGET_COL = params["target_col"]
    N_FORECAST = params["n_forecast"]
    INPUT_FILE = params["input_file"]
    TRAIN_START = params["train_start"]
    TRAIN_END = params["train_end"]
    PREPROCESSOR_FILE = params["preprocessor_file"]
    METRICS_FILE = params["metrics_file"]
    MODEL_NAME = params["model_name"]


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
    print("multi-step output created")
    X_train, y_train, X_test, y_test = create_sets(
        df,
        dt_col=DT_COL,
        target_col=TARGET_COL,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
    )
    print("train test split created")
    X_train_transformed, X_test_transformed = fe_transform(
        X_train,
        X_test,
        preprocessor_file=PREPROCESSOR_FILE,
    )

    best_params = train_model(
            X_train_transformed=X_train_transformed,
            y_train=y_train,
        )
    
    final_train(
            best_params=best_params,
            X_train=X_train_transformed,
            y_train=y_train,
            X_test=X_test_transformed,
            y_test=y_test,
            model_name=MODEL_NAME,
            metrics_file=METRICS_FILE,
        )
