import os
import hsfs
import numpy as np
import pandas as pd
import joblib
from typing import Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from amazonstockprediction.utils import (
    get_feature_store,
    get_train_test_val_dates,
    generate_sequence,
)
from amazonstockprediction.logger import setup_logger

# Setup logger
logger = setup_logger("training_pipeline")


def get_or_create_feature_view(
    feature_store: FeatureStore,
) -> Tuple[FeatureView, str]:
    """Create a feature view in Hopsworks from the selected features in the provided feature group."""
    try:
        amazon_fg = feature_store.get_feature_group("amazon_stock_prices", version=1)
        selected_features = amazon_fg.select(
            ["datetime", "open", "high", "close", "low", "volume", "rsi", "cci"]
        )

        start_date = pd.DataFrame(amazon_fg.show(1))["id"][0].split()[0]

        amazon_fv = feature_store.get_or_create_feature_view(
            name="amazon_fv",
            version=1,
            query=selected_features,
        )
        return amazon_fv, start_date
    except Exception as e:
        logger.error(f"Error creating feature view: {e}")
        raise

def get_time_series_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    window_size: int = 28,
    forecast_steps: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into train, val and test sets and generate sequences."""
    try:

        # Drop the date time column
        train = train.drop("datetime", axis=1)
        val = val.drop("datetime", axis=1)
        test = test.drop("datetime", axis=1)

        X_train, y_train = generate_sequence(
            train, window_size=window_size, forecast_steps=forecast_steps
        )
        X_val, y_val = generate_sequence(
            val, window_size=window_size, forecast_steps=forecast_steps
        )
        X_test, y_test = generate_sequence(
            test, window_size=window_size, forecast_steps=forecast_steps
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        logger.error(f"Error generating time series data: {e}")
        raise

def train_and_evaluate_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[Any, float]:
    try:
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], -1)

        model = XGBRegressor(objective="reg:squarederror")

        model.fit(X_train_reshaped, y_train)

        y_pred_val = model.predict(X_val_reshaped)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        logger.info(f"Validation RMSE: {val_rmse}")

        return model, val_rmse
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise

def save_model(model: BaseEstimator, model_dir: str, model_path: str):
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    try:
        fs, mr = get_feature_store()
        amazon_fv, start_date = get_or_create_feature_view(fs)

        (
            train_start_dt,
            train_end_dt,
            val_start_dt,
            val_end_dt,
            test_start_dt,
            test_end_dt,
        ) = get_train_test_val_dates(start_date)

        train, val, test, _, _, _ = amazon_fv.train_validation_test_split(
            train_start=train_start_dt,
            train_end=train_end_dt,
            val_start=val_start_dt,
            val_end=val_end_dt,
            test_start=test_start_dt,
            test_end=test_end_dt,
        )

        X_train, y_train, X_val, y_val, X_test, y_test = get_time_series_data(
            train, val, test, window_size=28, forecast_steps=7
        )

        best_model, val_rmse = train_and_evaluate_model(X_train, y_train, X_val, y_val)

        model_dir = "../models/xgboost_model"
        model_path = os.path.join(model_dir, "xgboost_model.pkl")
        save_model(best_model, model_dir, model_path)
        metrics = {"rmse": val_rmse}

        model = mr.python.create_model(
            name="amazon_stock_price_prediction_model_xgboost",
            description="XGBoost model for predicting Amazon stock prices",
            input_example=X_train[0],
            feature_view=amazon_fv,
            metrics = metrics
        )
        model.save(model_dir)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
