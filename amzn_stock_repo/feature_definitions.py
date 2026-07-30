from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, Project, PushSource, ValueType
from feast.types import Float64

# Define a project for the feature repo
project = Project(
    name="amzn_stock_repo",
    description="a project for predicting amazon stock prices",
)

# Define an entity
stock = Entity(name="stock", join_keys=["ticker"], value_type=ValueType.STRING)

# Define a file source
amazon_stock_source = FileSource(
    name="amazon_stock_source",
    path="data/amazon_stock_features.parquet",
    timestamp_field="datetime",
)

# stock feature push source
amazon_stock_push_source = PushSource(
    name="amazon_stock_push_source",
    batch_source=amazon_stock_source
)

# create a feature view
amazon_stock_fv = FeatureView(
    name="amazon_stock_fv",
    entities=[stock],
    ttl=timedelta(hours=1),
    schema=[
        Field(
            name="close", dtype=Float64
        ),
        Field(name='high', dtype= Float64),
        Field(name="low", dtype=Float64),
        Field(name="open", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="rsi", dtype=Float64),
        Field(name="cci", dtype=Float64)
        
    ],
    online=True,
    source=amazon_stock_push_source,
)
