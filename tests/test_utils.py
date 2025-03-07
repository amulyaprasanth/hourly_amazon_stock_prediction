import pytest
import pandas as pd
from amazonstockprediction import utils



def test_fetch_historical_data():
    data = utils.fetch_historical_data(ticker="AMZN", period="2y", interval="1h")
    assert isinstance(data, pd.DataFrame)
    assert "datetime" in data.columns
    assert "id" in data.columns


def test_calculate_indicators():
    data = pd.DataFrame(
        {
            "close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "high": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "low": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    data = utils.calculate_indicators(data)
    assert "rsi" in data.columns
    assert "cci" in data.columns


def test_generate_sequence():
    data = pd.DataFrame(
        {
            "close": [i for i in range(100)],
            "high": [i for i in range(100)],
            "low": [i for i in range(100)],
        }
    )
    window_size = 24
    forecast_steps = 6
    X, y = utils.generate_sequence(data, window_size, forecast_steps)
    assert len(X) == len(data) - window_size - forecast_steps
    assert len(y) == len(data) - window_size - forecast_steps
