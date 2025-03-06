import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator

def fetch_historical_data(ticker: str = "AMZN", period: str = "2y", interval: str = "1h") -> pd.DataFrame: 
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
    data = pd.DataFrame(yf.download(tickers=ticker, period=period, interval=interval, multi_level_index=False))

    # Reset the index to convert the date index into a column
    data = data.reset_index()

    
    # Remove the time zone information from the 'Datetime' column
    data['Datetime'] = pd.to_datetime(data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'))

    # Rename columns to lower case for consistency
    data.columns = [column.lower() for column in data.columns]

    # Add the 'id' column as a primary key, which is a string representation of the 'datetime' column
    data["id"] = [str(date) for date in data['datetime']]
    
    return data



def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates RSI and CCI indicators for the given stock data.

    Args:
        data (pd.DataFrame): The stock data.

    Returns:
        pd.DataFrame: The stock data with RSI and CCI indicators.
    """
    rsi = RSIIndicator(data['close']).rsi()
    cci = CCIIndicator(data['high'], data['low'], data['close']).cci()
    data['rsi'] = rsi
    data['cci'] = cci

    return data.dropna()