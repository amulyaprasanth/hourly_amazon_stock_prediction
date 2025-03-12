import os
import joblib
import torch
import datetime
import numpy as np
from hsfs.feature_view import FeatureView
import pandas as pd
from sklearn.preprocessing import StandardScaler
from amazonstockprediction.logger import setup_logger
from amazonstockprediction.utils import get_feature_store, get_and_download_best_model, read_yaml, generate_sequence
from amazonstockprediction.training_pipeline import LSTMModel
# Setup logger
logger = setup_logger("inference_pipeline")

# setup ocnfig
config = read_yaml()

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
    amazon_fv = fs.get_feature_view(config['data_params']['feature_view_name'], version=1)

    return fs, mr, amazon_fv


def prepare_inference_data(fv: FeatureView, preprocessor: StandardScaler, window_size: int = config['data_params']['window_size']):
    try:
        logger.info("Preparing inference data...")
        # Get the last window_size data points from the feature view
        last_batch_data = (
            fv.get_batch_data().sort_values("datetime").iloc[-window_size:, :]
        )

        # Reshape the data for inference to feed it into the model
        last_batch_data_reshaped = np.expand_dims(
            preprocessor.transform(last_batch_data.drop("datetime", axis=1)), axis=0
        )


        # Validate the shape of the data
        assert (
            last_batch_data_reshaped.shape == (1, 28, 7)
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
        {"datetime": time_index, "close": predictions.squeeze()}
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
            model_name=config["model_params"]['lstm_model']["model_name"],
            model_dir=config['model_params']['lstm_model']['model_dir'],
        )

        # Load model
        logger.info("Loading model and preprocessor...")
        model = LSTMModel(
           input_size = config["model_params"]["lstm_model"]["input_size"],
           hidden_size = config["model_params"]["lstm_model"]["hidden_size"],
           num_layers = config["model_params"]["lstm_model"]["num_layers"],
           output_size= config["data_params"]["forecast_steps"]
        )
        
        model = torch.load(os.path.join(download_dir, config['model_params']['lstm_model']['model_filename']), weights_only= False)
        preprocessor = joblib.load(os.path.join(download_dir, config['model_params']['preprocessor_filename']))


        # Prepare inference data
        inference_data = prepare_inference_data(amazon_fv, preprocessor)
        
        # Make predictions
        logger.info("Making predictions...")
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(inference_data, dtype=torch.float32))


        # Inverse transform the predictions
        feature_mean = preprocessor.mean_[2] # type: ignore
        feature_std = preprocessor.scale_[2] # type: ignore
        
        # Inverse transform the scaled value
        inverse_transformed_preds = predictions * feature_std + feature_mean

        # Create a dataframe from the predictions
        logger.info("Creating predictions dataframe...")
        predictions_df = create_dataframe_from_predictions(inverse_transformed_preds)

        # Create an id column to act as primary key in our feature group
        predictions_df["id"] = predictions_df["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        logger.info("Inserting predictions into feature store...")
        amazon_prediction_fg = fs.get_or_create_feature_group(
            name=config["inference_params"]["predictions_feature_group"],
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
