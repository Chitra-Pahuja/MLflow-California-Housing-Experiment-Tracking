"""
California Housing Price Prediction with Gradient Boosting
Mirrors the lab's linear_regression.py structure but uses:
- Dataset: California Housing (sklearn) instead of Wine Quality
- Model: GradientBoostingRegressor instead of ElasticNet
- Params: n_estimators, max_depth, learning_rate instead of alpha, l1_ratio

Usage:
    python housing_prediction.py <n_estimators> <max_depth> <learning_rate>
    python housing_prediction.py 100 5 0.1
    python housing_prediction.py  (uses defaults: 100, 3, 0.1)
"""

import logging
import os
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Fix for paths with spaces — use SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame  # includes target column 'MedHouseVal'

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "MedHouseVal" (median house value in $100k)
    train_x = train.drop(["MedHouseVal"], axis=1)
    test_x = test.drop(["MedHouseVal"], axis=1)
    train_y = train[["MedHouseVal"]]
    test_y = test[["MedHouseVal"]]

    # Hyperparameters from command line or defaults
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    with mlflow.start_run():
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        gb.fit(train_x, train_y.values.ravel())

        predicted_values = gb.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_values)

        print(
            f"GradientBoosting model (n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate:f}):"
        )
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = gb.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                gb,
                "model",
                registered_model_name="GradientBoostingHousingModel",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(gb, "model", signature=signature)