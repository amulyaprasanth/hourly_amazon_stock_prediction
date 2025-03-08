import os
import hopsworks

from hsml.model_registry import ModelRegistry
from hsfs.feature_store import FeatureStore
import yfinance as yf
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from typing import Tuple
from amazonstockprediction.logger import setup_logger

load_dotenv()

# Configure logging
logger = setup_logger("helper_functions_logger")

# Get the environment variables
hopsworks_api_key = str(os.getenv("HOPSWORKS_API_KEY"))


def get_feature_store(
    api_key_value: str = hopsworks_api_key) -> Tuple[FeatureStore, ModelRegistry]:
    """Login to hopsworks and get feature store.
    Args:
        api_key_value (str): Hopsworks API key.
        model_registry (bool): Flag to indicate if model registry is needed.

    Returns:
        Union[FeatureStore, Tuple[FeatureStore, ModelRegistry]]: The feature store object or a tuple with feature store and model registry.
    """
    try:
        project = hopsworks.login(api_key_value=api_key_value)
        fs = project.get_feature_store()
        mr = project.get_model_registry()

        return fs, mr
    except Exception as e:
        logger.error(f"Failed to get feature store: {e}")
        raise


def fetch_historical_data(
    ticker: str = "AMZN", period: str = "2y", interval: str = "1h"
) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker, period, and interval.

    Parameters:
    ticker (str): The stock ticker symbol (default is "AMZN").
    period (str): The period over which to fetch data (default is "2y").
    interval (str): The interval between data points (default is "1h").

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data with the following modifications:
        - Index reset to convert the date index into a column.
        - Time zone information removed from the 'Datetime' column.
        - Column names converted to lower case.
        - An 'id' column added as a primary key, which is a string representation of the 'datetime' column.
    """
    try:
        data = pd.DataFrame(
            yf.download(
                tickers=ticker,
                period=period,
                end=datetime.now() - timedelta(days=1), # Get the historical data upto yesterday
                interval=interval,
                multi_level_index=False,
            )
        )
        data = data.reset_index()
        data["Datetime"] = pd.to_datetime(
            data["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        data.columns = [column.lower() for column in data.columns]
        data["id"] = [str(date) for date in data["datetime"]]
        return data
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        raise


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates RSI and CCI indicators for the given stock data.

    Args:
        data (pd.DataFrame): The stock data.

    Returns:
        pd.DataFrame: The stock data with RSI and CCI indicators.
    """
    try:
        rsi = RSIIndicator(data["close"]).rsi()
        cci = CCIIndicator(data["high"], data["low"], data["close"]).cci()
        data["rsi"] = rsi
        data["cci"] = cci
        return data.dropna()
    except Exception as e:
        logger.error(f"Failed to calculate indicators: {e}")
        raise


def get_train_test_val_dates(
    start_date: str,
    end_date: str = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d"),
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[str, str, str, str, str, str]:
    """Split the data into training, validation, and test sets.
    Arguments:
        start_date (str): The start date of the data.
        end_date (str): The end date of the data.
        train_size (float): The proportion of data for training.
        val_size (float): The proportion of data for validation.

    Returns:
        tuple: The start and end dates for training, validation, and test sets.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        train_days = int(train_size * total_days)
        val_days = int(val_size * total_days)
        train_start_dt = start_dt
        train_end_dt = start_dt + timedelta(days=train_days - 1)
        val_start_dt = train_end_dt + timedelta(days=1)
        val_end_dt = val_start_dt + timedelta(days=val_days - 1)
        test_start_dt = val_end_dt + timedelta(days=1)
        test_end_dt = end_dt
        return (
            train_start_dt.strftime("%Y-%m-%d"),
            train_end_dt.strftime("%Y-%m-%d"),
            val_start_dt.strftime("%Y-%m-%d"),
            val_end_dt.strftime("%Y-%m-%d"),
            test_start_dt.strftime("%Y-%m-%d"),
            test_end_dt.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        logger.error(f"Failed to split data into train, validation, and test sets: {e}")
        raise


def generate_sequence(
    data: pd.DataFrame, window_size: int = 24, forecast_steps: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences of data for time series forecasting.

    Args:
        data (pd.DataFrame): The stock data.
        window_size (int): The size of the window for input sequences.
        forecast_steps (int): The number of steps to forecast.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of input sequences and corresponding outputs.
    """
    try:
        X = []
        y = []
        for i in range(len(data) - window_size - forecast_steps):
            X.append(data.iloc[i : i + window_size].values)
            y.append(
                data.iloc[i + window_size : i + window_size + forecast_steps, 2].values
            )
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Failed to generate sequences: {e}")
        raise


def get_and_download_best_model(mr: ModelRegistry, model_name: str, model_dir: str) -> str:
    """
    Download the best model from the model registry.

    Args:
        mr (ModelRegistry): The model registry object.
        model_name (str): The name of the model to download.
        model_dir (str): The directory where the model should be saved.

    Returns:
        str: The path to the downloaded model.

    Raises:
        Exception: If there is an error in downloading the model.
    """
    try:
        EVALUATION_METRIC = "rmse"
        SORT_METRICS_BY = "min"

        # Get the best model from the model registry
        best_model = mr.get_best_model(model_name, EVALUATION_METRIC, SORT_METRICS_BY)

        # Ensure the model directory exists
        if not os.path.exists(model_dir):
            logger.info(f"Model directory doesn't exist, creating directory: {model_dir}")
            os.makedirs(model_dir)

        # Download the model
        if best_model is not None:
            best_model_path = best_model.download(model_dir)
            return best_model_path
        else:
            raise Exception("Best model not found in the model registry.")

    except Exception as e:
        logger.error(f"Failed to download the best model: {e}")
        raise
