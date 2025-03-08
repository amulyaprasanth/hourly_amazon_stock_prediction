import os
import joblib
import datetime
import numpy as np
from hsfs.feature_view import FeatureView
import pandas as pd
from amazonstockprediction.logger import setup_logger
from amazonstockprediction.utils import get_feature_store, get_and_download_best_model

# Setup logger
logger = setup_logger("inference_pipeline")


def get_hopsworks_objects():
    """Returns feature store, model registry and Amazon feature view.
    Returns:
    fs (FeatureStore): The feature store object.
    mr (ModelRegistry): The model registry object.
    amazon_fv (FeatureView): The Amazon feature view object."""

    # Get the feature store and model registry
    logger.info("Getting feature store and model registry...")
    fs, mr = get_feature_store()

    logger.info("Getting Amazon feature view...")
    amazon_fv = fs.get_feature_view("amazon_fv", version=1)

    return fs, mr, amazon_fv


def prepare_inference_data(fv: FeatureView, window_size: int = 28):
    try:
        logger.info("Preparing inference data...")
        # Get the last window_size data points from the feature view
        last_batch_data = (
            fv.get_batch_data().sort_values("datetime").iloc[-window_size:, :]
        )

        # Reshape the data for inference to feed it into the model
        last_batch_data_reshaped = np.expand_dims(
            last_batch_data.drop("datetime", axis=1), axis=0
        ).reshape(1, -1)

        # Validate the shape of the data
        assert (
            last_batch_data_reshaped.shape[1] == 28 * 7
        ), "The data shape does not match the expected input shape for the model"

        logger.info("Inference data prepared successfully.")
        return last_batch_data_reshaped

    except Exception as e:
        logger.error(f"Failed to prepare inference data: {e}")
        raise


def create_dataframe_from_predictions(predictions: np.ndarray):
    logger.info("Creating dataframe from predictions...")
    time_index = [
        "14:30:00",
        "15:30:00",
        "16:30:00",
        "17:30:00",
        "18:30:00",
        "19:30:00",
        "20:30:00",
    ]

    # Add today's date to the time index
    today = datetime.date.today().strftime("%Y-%m-%d")
    time_index = [f"{today} {t}" for t in time_index]

    predictions_df = pd.DataFrame(
        {"datetime": time_index, "prediction": predictions.squeeze()}
    )

    predictions_df["datetime"] = pd.to_datetime(predictions_df["datetime"])

    logger.info("Dataframe created successfully.")
    return predictions_df


if __name__ == "__main__":
    try:
        logger.info("Starting batch inference pipeline...")

        # Get Hopsworks objects
        fs, mr, amazon_fv = get_hopsworks_objects()

        # Get and download the best model based on RMSE
        logger.info("Downloading the best model...")
        download_dir = get_and_download_best_model(
            mr,
            model_name="amazon_stock_price_prediction_model_xgboost",
            model_dir="../models/xgboost_model",
        )

        # Load model
        logger.info("Loading model...")
        model = joblib.load(os.path.join(download_dir, "xgboost_model.pkl"))

        # Prepare inference data
        inference_data = prepare_inference_data(amazon_fv)

        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(inference_data)

        # Create a dataframe from the predictions
        logger.info("Creating predictions dataframe...")
        predictions_df = create_dataframe_from_predictions(predictions)

        # Create an id column to act as primary key in our feature group
        predictions_df["id"] = predictions_df["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        logger.info("Inserting predictions into feature store...")
        amazon_prediction_fg = fs.get_or_create_feature_group(
            name="amazon_stock_predictions",
            description="Amazon stock predictions",
            version=1,
            online_enabled=True,
            primary_key=["id"],
            event_time="datetime",
        )

        amazon_prediction_fg.insert(predictions_df)
        logger.info("Batch inference pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Batch inference pipeline failed: {e}")
        raise
