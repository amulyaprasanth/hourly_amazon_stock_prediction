from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from feast.data_source import PushMode

from src import setup_logger
from src.utils import calculate_indicators, fetch_historical_data

# setup logger
logger = setup_logger("feature_pipeline")


class FeaturePipeline:
    """
    Pipeline responsible for fetching, preprocessing, and pushing
    yesterday's AMZN stock data into the Feast feature store.
    """

    def __init__(self):
        """
        Initialize the feature pipeline and connect to the Feast
        feature store repo.
        """
        try:
            self.feature_store = FeatureStore("amzn_stock_repo")
        except Exception as e:
            logger.error(f"Failed to initialize FeatureStore: {e}")
            raise

    def get_yesterdays_data(self) -> tuple[pd.DataFrame, str]:
        """
        Fetches yesterday's stock data, with enough preceding history
        for rolling-window indicators (RSI, CCI) to be valid.
        """
        try:
            yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
            logger.info(f"Fetching data for {yesterday}")

            data = fetch_historical_data(period="30d", interval="1h")

            data["datetime"] = pd.to_datetime(data["datetime"])

            return data, yesterday

        except Exception as e:
            logger.error(f"Failed to fetch yesterday's data: {e}")
            raise

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw stock data before pushing to the feature store.

        Adds a ticker column, computes technical indicators, and
        updates the entity parquet file with any new (ticker, datetime)
        rows, deduplicating against existing entries.

        Args:
            data (pd.DataFrame): Raw stock data (e.g. from
                `get_yesterdays_data`).

        Returns:
            pd.DataFrame: Preprocessed data with indicators added.

        Raises:
            FileNotFoundError: If the existing entity parquet file
                cannot be found.
            Exception: If any other step of preprocessing fails.
        """
        try:
            # calculate the indicators first
            data = calculate_indicators(data)
            logger.info("Calculated technical indicators")

            # add ticker column after indicators, so it can't be lost/misaligned
            data["ticker"] = "AMZN"
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            raise

        try:
            # update the entity parquet file
            new_entity_df = data[["ticker", "datetime"]]
            old_entity_df = pd.read_parquet("amzn_stock_repo/data/entity_df.parquet")
            entity_df = pd.concat(
                [old_entity_df, new_entity_df],
                ignore_index=True,
            ).drop_duplicates()
            entity_df.to_parquet("amzn_stock_repo/data/entity_df.parquet")
            logger.info(f"Updated entity parquet file, {len(entity_df)} total rows")

        except FileNotFoundError as e:
            logger.error(f"Couldn't find entity parquet file: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to update entity parquet file: {e}")
            raise

        return data

    def run(self):
        logger.info("Starting feature pipeline run")

        full_history, yesterday = self.get_yesterdays_data()

        # compute indicators
        full_history = self.preprocess(full_history)

        # filter to yesterday's rows
        data = full_history[
            full_history["datetime"].dt.date == pd.to_datetime(yesterday).date()
        ].reset_index(drop=True)

        if data.empty:
            logger.warning("No data to process, skipping push to feature store")
            return

        if data[["rsi", "cci"]].isnull().any().any():
            logger.warning("rsi/cci contain nulls even after using extended history")

        data = data[
            [
                "datetime",
                "close",
                "high",
                "low",
                "open",
                "volume",
                "ticker",
                "rsi",
                "cci",
            ]
        ].reset_index(drop=True)
        data["datetime"] = data["datetime"].astype("datetime64[us]")

        try:
            self.feature_store.push(
                "amazon_stock_push_source", data, to=PushMode.ONLINE_AND_OFFLINE
            )
            logger.info(
                f"Pushed {len(data)} rows to feature store (online and offline)"
            )
        except Exception as e:
            logger.error(f"Failed to push data to feature store: {e}")
            raise
        
        
        prediction_path = Path("artifacts/predictions.csv")

        if prediction_path.exists():
            predictions = pd.read_csv(prediction_path)
            predictions["datetime"] = pd.to_datetime(predictions["datetime"])

            actual = data[["datetime", "close"]].rename(
                columns={"close": "actual"}
            )

            prediction_vs_actual = predictions.merge(
                actual,
                on="datetime",
                how="inner",
            )

            prediction_vs_actual.to_csv(
                "artifacts/predicted_vs_actual.csv",
                index=False,
            )

            logger.info(
                "Saved %d prediction/actual pairs.",
                len(prediction_vs_actual),
            )
        else:
            logger.warning("predictions.csv not found. Skipping evaluation file.")

        logger.info("Feature pipeline run completed successfully")


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
