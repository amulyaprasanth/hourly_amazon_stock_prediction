import pandas as pd

from src import setup_logger
from src.utils import calculate_indicators, fetch_historical_data

logger = setup_logger("backfill_features")


class HistoricalFeatures:
    def __init__(self, ticker: str, time_period: str, interval: str):
        self.ticker = ticker
        self.time_period = time_period
        self.interval = interval

        self.df = self.get_and_process_historical_data()
        
        # entity df should contain ticker which is an entity here, timestamp and label col
        self.entity_df = self.create_entity_df(self.df)

        self.save_to_parquet(
            self.df,
            "amzn_stock_repo/data/amazon_stock_features.parquet",
        )

        self.save_to_parquet(
            self.entity_df,
            "amzn_stock_repo/data/entity_df.parquet",
        )

    def get_and_process_historical_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data, calculate technical indicators,
        and prepare it for the Feast feature store.
        """
        try:
            logger.info("Fetching historical data...")
            df = fetch_historical_data(
                self.ticker,
                self.time_period,
                self.interval,
            )

            # Feast requires an entity column.
            df["ticker"] = self.ticker

            logger.info("Calculating indicators...")
            df = calculate_indicators(df)

            return df

        except Exception:
            logger.exception("Failed to fetch and process historical data.")
            raise

    def create_entity_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the entity DataFrame required by Feast for
        historical feature retrieval.
        """

        entity_df = df.copy()

        # Feast only needs entity, timestamp and label
        entity_df = entity_df[
            [
                "ticker",
                "datetime",
            ]
        ]

        return entity_df

    def save_to_parquet(self, df: pd.DataFrame, save_filepath: str):
        """
        Save a DataFrame as a Parquet file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            save_filepath (str): Destination Parquet file path.
        """
        logger.info(f"Saving parquet file to {save_filepath}")
        df.to_parquet(save_filepath, index=False)


if __name__ == "__main__":
    historical_features = HistoricalFeatures("AMZN", "2y", "1h")

    print("Feature DataFrame")
    print(historical_features.df.head())

    print("\nEntity DataFrame")
    print(historical_features.entity_df.head())
