import os

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator

from src import setup_logger

load_dotenv()

# Configure logging
logger = setup_logger("helper_functions_logger")

# Get the environment variables
hopsworks_api_key = str(os.getenv("HOPSWORKS_API_KEY"))


def read_yaml(yaml_file_path: str = "config/config.yml"):
    """Reads a YAML file and returns the contents as a dictionary.
    Args:
        yaml_file_path (str): The path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    Raises:
        Exception: If there is an error in reading the YAML file."""
    with open(yaml_file_path, "r") as f:
        try:
            data = yaml.safe_load(f)
            return data

        except yaml.YAMLError as e:
            logger.error(f"Failed to read YAML file: {e}")
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
                interval=interval,
                multi_level_index=False,
            )
        )
        data = data.reset_index()
        data["Datetime"] = pd.to_datetime(
            data["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        data.columns = [column.lower() for column in data.columns]
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


def generate_sequences(
    data: np.ndarray,
    target_index: int,
    window_size: int = 24,
    forecast_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate input/output sequences for multi-step time series forecasting.

    Args:
        data: NumPy array of shape (num_samples, num_features).
        target_index: Index of the target column.
        window_size: Number of historical timesteps used as input.
        forecast_steps: Number of future timesteps to predict.

    Returns:
        Tuple containing:
            X: Shape (samples, window_size, num_features)
            y: Shape (samples, forecast_steps)
    """
    try:
        X = []
        y = []

        num_rows = len(data)

        for i in range(num_rows - window_size - forecast_steps + 1):
            X.append(data[i : i + window_size])

            y.append(
                data[
                    i + window_size : i + window_size + forecast_steps,
                    target_index,
                ]
            )

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )

    except Exception:
        logger.exception("Failed to generate sequences.")
        raise
