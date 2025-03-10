import os
import requests
import hopsworks
import pandas as pd
from amazonstockprediction.logger import setup_logger
from amazonstockprediction.utils import fetch_historical_data, calculate_indicators, read_yaml
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the data params config
data_params = read_yaml()["data_params"]

# Get the environment variables
hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
logger = setup_logger("backfill_features")

def get_and_process_data():
    """Fetch historical stock data for Amazon and calculate indicators."""
    try:
        # Fetch data from API
        logger.info("Fetching historical data...")
        df = fetch_historical_data("AMZN", "2y", "1h")

        # Calculate indicators
        logger.info("Calculating indicators...")
        df = calculate_indicators(df)
        return df

    except requests.exceptions.RequestException as re:
        logger.error(f"RequestException: {re}")
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None

def upload_to_feature_store(df):
    """Upload processed data to Hopsworks Feature Store."""
    if df is None or df.empty:
        logger.error("Dataframe is empty or None. Skipping upload.")
        return

    try:
        # Connect to Hopsworks Feature Store
        project = hopsworks.login(api_key_value=str(hopsworks_api_key))
        fs = project.get_feature_store()

        # Upload data to Feature Store
        logger.info("Uploading data to Hopsworks Feature Store...")

        # Get or Create feature group
        amazon_fg = fs.get_or_create_feature_group(
            name=data_params['feature_group_name'],
            description="Amazon last 5 year stock prices",
            version=1,
            online_enabled=True,
            primary_key=["id"],
            event_time="datetime"
        )
        logger.info("Feature group created or retrieved.")

        # Upload data
        amazon_fg.insert(df)
        logger.info("Data uploaded successfully.")

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except hopsworks.client.exceptions.RestAPIError as re:
        logger.error(f"Hopsworks API error: {re}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

def main():
    # Get and process data
    df = get_and_process_data()

    # Upload to feature store
    upload_to_feature_store(df)

if __name__ == "__main__":
    main()