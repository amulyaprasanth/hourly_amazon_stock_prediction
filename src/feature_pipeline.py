from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import sessionmaker
from tqdm.auto import tqdm

from src import setup_logger
from src.db import Feature, engine
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
            Session = sessionmaker(engine)
            self.session = Session()
        except Exception:
            logger.exception("Unable to connect to the database.")
            raise

    def get_yesterdays_data(self) -> tuple[pd.DataFrame, str]:
        """
        Fetches yesterday's stock data, with enough preceding history
        for rolling-window indicators (RSI, CCI) to be valid.
        """
        try:
            ET = ZoneInfo("America/New_York")
            yesterday = (datetime.now(ET).date() - timedelta(days=1)).isoformat()
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

        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            raise

        return data

    def run(self):
        logger.info("Starting feature pipeline run")

        full_history, yesterday = self.get_yesterdays_data()

        # compute indicators
        full_history = self.preprocess(full_history)

        # filter to yesterday's rows
        data: pd.DataFrame = full_history.loc[
            full_history["datetime"].dt.date == pd.to_datetime(yesterday).date()
        ].reset_index(drop=True)

        if data.empty:
            logger.warning("No data to process, skipping push to feature store")
            return

        for _, row in tqdm(
            data.iterrows(), total=len(data), desc="updating database with latest data"
        ):
            user_record = Feature(
                id=row["datetime"],
                close=row["close"],
                high=row["high"],
                low=row["low"],
                open=row["open"],
                volume=row["volume"],
                rsi=row["rsi"],
                cci=row["cci"],
            )
            self.session.add(user_record)

        self.session.commit()
        self.session.close()

        # Adding actual values to the predictions vs actual dataframe
        prediction_path = Path("artifacts/predictions.csv")

        if prediction_path.exists():
            predictions = pd.read_csv(prediction_path)
            predictions["datetime"] = pd.to_datetime(predictions["datetime"])

            # use the date from predictions to avoid empty today_data
            # Both yfinance data and forecast timestamps are in naive ET,
            # so derive today's date in ET to avoid UTC/ET date boundary issues.
            day = predictions["datetime"].iloc[0].date()
            today_data = full_history.loc[
                full_history["datetime"].dt.date == day
            ].reset_index(drop=True)

            if today_data.empty:
                logger.warning(
                    "No actual data found for %s. Skipping evaluation file.", day
                )
            else:
                actual = today_data[["datetime", "close"]].rename(
                    columns={"close": "actual"}
                )
                actual["datetime"] = pd.to_datetime(actual["datetime"]).dt.tz_localize(
                    None
                )
                predictions["datetime"] = predictions["datetime"].dt.tz_localize(None)

                prediction_vs_actual = predictions.merge(
                    actual, on="datetime", how="inner"
                )

                prediction_vs_actual.to_csv(
                    "artifacts/predicted_vs_actual.csv",
                    index=False,
                )

                logger.info(
                    "Saved %d prediction/actual pairs.",
                    len(prediction_vs_actual),
                )

        logger.info("Feature pipeline run completed successfully")


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
