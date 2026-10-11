from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

# Create an engine
engine = create_engine(
    "sqlite:///features.db",
    echo=True,
)

Base = declarative_base()


class Feature(Base):
    __tablename__ = "features"
    id: Mapped[datetime] = mapped_column(primary_key=True)
    close: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    open: Mapped[float]
    volume: Mapped[float]
    rsi: Mapped[float]
    cci: Mapped[float]

    def __repr__(self):
        return f"Feature(id={self.id!r}, close={self.close!r}, high={self.high!r}, low={self.low!r}, open={self.open!r}, volume={self.volume}, rsi={self.rsi!r}, cci={self.cci!r})"


Base.metadata.create_all(engine)

if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy.orm import Session
    from sqlalchemy import func, select

    session = Session(engine)
    df = pd.read_parquet("amzn_stock_repo/data/amazon_stock_features.parquet")
    row = df.iloc[0]
    feature = Feature(
        id=row["datetime"],
        close=row["close"],
        high=row["high"],
        low=row["low"],
        open=row["open"],
        volume=row["volume"],
        rsi=row["rsi"],
        cci=row["cci"],
    )

    # session.add(feature)
    query = select(func.count(Feature.id))

    total_rows = session.scalars(query)
    print(total_rows.first())
