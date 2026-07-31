import json

import joblib
import numpy as np
import pandas as pd
import torch

from src import setup_logger
from src.models.lstm import LSTMModel
from src.utils import calculate_indicators, fetch_historical_data, read_yaml

# setup logger
logger = setup_logger("inference_pipeline")

# read the config file
config = read_yaml()


class InferencePipeline:
    """
    Pipeline responsible for generating stock price forecasts using
    the current MLflow "champion" model and the fitted scaler from
    training.
    """

    def __init__(self):
        """
        Initialize the inference pipeline: load config, feature
        columns, the fitted scaler, and the champion model from the
        MLflow Model Registry.
        """
        self.feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "cci",
        ]
        self.target_index = self.feature_columns.index("close")

        self.window_size = config["data_params"]["window_size"]
        self.forecast_steps = config["data_params"]["forecast_steps"]

        self.model_name = "amazon-stock-lstm"
        self.model_alias = "champion"
        self.scaler_path = "artifacts/scaler.pkl"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_size = config["model_params"]["lstm_model"]["input_size"]
        self.hidden_size = config["model_params"]["lstm_model"]["hidden_size"]
        self.num_layers = config["model_params"]["lstm_model"]["num_layers"]
        self.output_size = config["data_params"]["forecast_steps"]
        self.scaler = self._load_scaler()
        self.model = LSTMModel(
            self.input_size, self.hidden_size, self.num_layers, self.output_size
        )
        self.registry_path = "src/models/registry.json"
        self.load_best_model()
        

    def _load_scaler(self):
        """
        Load the fitted StandardScaler saved during training.

        Returns:
            StandardScaler: The fitted scaler.

        Raises:
            FileNotFoundError: If the scaler file doesn't exist.
            Exception: If loading fails for any other reason.
        """
        try:
            scaler = joblib.load(self.scaler_path)
            logger.info(f"Loaded scaler from {self.scaler_path}")
            return scaler
        except FileNotFoundError as e:
            logger.error(f"Scaler file not found at {self.scaler_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            raise

    def get_latest_features(self) -> pd.DataFrame:
        """
        Fetch enough recent hourly data to build one input sequence,
        with a buffer for rolling-window indicators (RSI, CCI) to warm
        up before slicing to the last `window_size` rows.

        Returns:
            pd.DataFrame: DataFrame with the last `window_size` rows
                of feature-engineered data, ordered by datetime.

        Raises:
            Exception: If fetching or indicator calculation fails.
            ValueError: If fewer than `window_size` valid rows remain
                after computing indicators.
        """
        try:
            # fetch extra history so rolling-window indicators can warm up
            # before we slice down to the exact window_size we need
            data = fetch_historical_data(period="30d", interval="1h")
            data["datetime"] = pd.to_datetime(data["datetime"])
            data = data.sort_values("datetime").reset_index(drop=True)

            data = calculate_indicators(data)
            data = data.dropna(subset=self.feature_columns).reset_index(drop=True)

            if len(data) < self.window_size:
                raise ValueError(
                    f"Not enough valid rows ({len(data)}) to build a "
                    f"sequence of window_size={self.window_size}"
                )

            latest_window = data.iloc[-self.window_size :][self.feature_columns]
            logger.info(
                f"Fetched latest window of {len(latest_window)} rows for inference"
            )
            return latest_window

        except Exception as e:
            logger.error(f"Failed to fetch latest features: {e}")
            raise

    def preprocess(self, latest_window: pd.DataFrame) -> torch.Tensor:
        """
        Scale the latest feature window and convert it to a model-ready
        input tensor.

        Args:
            latest_window (pd.DataFrame): Last `window_size` rows of
                feature-engineered data.

        Returns:
            torch.Tensor: Input tensor of shape
                `(1, window_size, num_features)`.

        Raises:
            Exception: If scaling or tensor conversion fails.
        """
        try:
            scaled = self.scaler.transform(latest_window)
            x = np.expand_dims(scaled, axis=0)  # add batch dimension
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            return x_tensor
        except Exception as e:
            logger.error(f"Failed to preprocess input for inference: {e}")
            raise

    def postprocess(self, prediction: torch.Tensor) -> np.ndarray:
        """
        Inverse-transform the model's scaled prediction back to actual
        price values.

        The scaler was fit on all feature columns jointly, so each
        predicted (scaled) close value is placed into a dummy row
        matching the original feature shape before inverse-transforming,
        and only the target column is extracted afterward.

        Args:
            prediction (torch.Tensor): Model output of shape
                `(1, forecast_steps)`.

        Returns:
            np.ndarray: Array of shape `(forecast_steps,)` containing
                the forecasted close prices in original scale.

        Raises:
            Exception: If inverse transformation fails.
        """
        try:
            prediction_arr = prediction.detach().cpu().numpy().reshape(-1)

            # build dummy rows to inverse-transform just the target column
            dummy = np.zeros((len(prediction_arr), len(self.feature_columns)))
            dummy[:, self.target_index] = prediction_arr

            inverse_scaled = self.scaler.inverse_transform(dummy)
            forecasted_close = inverse_scaled[:, self.target_index]

            return forecasted_close
        except Exception as e:
            logger.error(f"Failed to inverse-transform prediction: {e}")
            raise

    def load_best_model(self):
        """
        Load the best-performing model from the local model registry.

        The registry maintains the top three models sorted in ascending
        order of RMSE, so the first entry is always considered the current
        best model. This method loads the corresponding model weights,
        moves the model to the configured device, and switches it to
        evaluation mode.

        Raises:
            RuntimeError:
                If the registry is empty, the model checkpoint cannot be
                loaded, or any other error occurs during the loading
                process.
        """
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)

            if len(registry) == 0:
                raise RuntimeError("No registered models found.")

            best_model = registry[0]

            state_dict = torch.load(
                best_model["path"],
                map_location=self.device,
                weights_only=True,
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "Loaded model %s (RMSE %.4f)",
                best_model["version"],
                best_model["rmse"],
            )

        except Exception as e:
            logger.exception("Failed to load best model.")
            raise RuntimeError(f"Could not load best model: {e}") from e

    def run(self) -> np.ndarray:
        """
        Run the full inference pipeline end-to-end.

        Fetches the latest feature window, scales it, runs the champion
        model's forward pass, and inverse-transforms the forecasted
        close prices back to their original scale.

        Returns:
            np.ndarray: Forecasted close prices for the next
                `forecast_steps` hours.
        """
        logger.info(
            f"Starting inference pipeline run using "
            f"'{self.model_name}@{self.model_alias}'"
        )

        latest_window = self.get_latest_features()
        x_tensor = self.preprocess(latest_window)

        try:
            with torch.no_grad():
                prediction = self.model(x_tensor)
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            raise

        forecast = self.postprocess(prediction)
        logger.info(f"Forecasted close prices: {forecast}")

        logger.info("Inference pipeline run completed successfully")
        return forecast


if __name__ == "__main__":
    try:
        logger.info("Starting inference pipeline...")
        pipeline = InferencePipeline()
        forecast = pipeline.run()
        logger.info("Inference pipeline completed successfully.")
        print(forecast)
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        raise
