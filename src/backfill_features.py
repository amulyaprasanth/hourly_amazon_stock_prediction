import pandas as pd
from sqlalchemy.orm import sessionmaker
from tqdm.auto import tqdm

from src import setup_logger
from src.db import Feature, engine
from src.utils import calculate_indicators, fetch_historical_data

logger = setup_logger("backfill_features")
Session = sessionmaker(engine)
session = Session()


class HistoricalFeatures:
    def __init__(self, ticker: str, time_period: str, interval: str):
        self.ticker = ticker
        self.time_period = time_period
        self.interval = interval

        self.df = self.get_and_process_historical_data()
        self.df["datetime"] = pd.to_datetime(
            self.df["datetime"], format="%Y-%m-%d %H:%M:%S"
        )

        logger.info("Adding historical data into db")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
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
            session.add(user_record)

        session.commit()
        session.close()

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

            logger.info(f"Found {len(df)} rows")

            logger.info("Calculating indicators...")
            df = calculate_indicators(df)

            return df

        except Exception:
            logger.exception("Failed to fetch and process historical data.")
            raise


if __name__ == "__main__":
    historical_features = HistoricalFeatures("AMZN", "2y", "1h")
