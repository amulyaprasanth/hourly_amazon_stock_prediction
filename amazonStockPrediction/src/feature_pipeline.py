import os
import hopsworks
from amazonStockPrediction.src.logger import setup_logger
from dotenv import load_dotenv
from amazonStockPrediction.src.utils import fetch_historical_data, calculate_indicators

load_dotenv()

# setup logger
logger = setup_logger("feature_pipeline")

try:
    # Get the environment variables
    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    if not hopsworks_api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable not set")

    # Login to project and feature store and get feature store
    project = hopsworks.login(api_key_value=str(hopsworks_api_key))
    fs = project.get_feature_store()
    amazon_fg = fs.get_feature_group("amazon_stock_prices", version=1)

    # Fetch historical stock data for Amazon
    logger.info("Fetching historical stock data for Amazon...")
    df = fetch_historical_data(period="20d", interval="1h")
    
    # Calculate indicators
    logger.info("Calculating RSI and CCI indicators...")
    df = calculate_indicators(df)

    # Inserting yesterday's data
    amazon_fg.insert(df.iloc[-7:, :])
    logger.info("Historical stock data for Amazon inserted successfully.")

except ValueError as ve:
    logger.error(f"ValueError: {ve}")
except hopsworks.client.exceptions.RestAPIError as re:
    logger.error(f"Hopsworks API error: {re}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")
