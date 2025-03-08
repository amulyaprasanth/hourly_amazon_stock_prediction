import os
import joblib
import numpy as np
from amazonstockprediction.logger import setup_logger
from amazonstockprediction.utils import get_feature_store


# Setup logger
logger = setup_logger("inference_pipeline")

def get_hopsworks_objects():
    """ Returns feature store, model registry and Amazon feature view. """

    # Get the feature store and model registry
    fs, mr = get_feature_store()
    amazon_fv = fs.get_feature_view("amazon_fv", version=1)

    return fs, mr, amazon_fv

## Download the best model from model registry
EVALUATION_METRIC = "rmse"
SORT_METRICS_BY = "min"

best_model = mr.get_best_model("amazon_stock_price_prediction_model_xgboost", EVALUATION_METRIC, SORT_METRICS_BY)

# Download the model
