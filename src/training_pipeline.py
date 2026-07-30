import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from feast import FeatureStore
from mlflow import exceptions
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src import setup_logger
from src.models.dataset import AmazonStockDataset
from src.models.lstm import LSTMModel
from src.models.trainer import Trainer
from src.utils import generate_sequences, read_yaml

# setup logger
logger = setup_logger("training_pipeline")

# read the config file
config = read_yaml()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# get the params config
class TrainingPipeline:
    def __init__(self):
        """
        Initialize the training pipeline.

        Sets up the Feast feature store connection, the feature scaler,
        and loads all data/model hyperparameters from the config file.
        """
        self.fs = FeatureStore("amzn_stock_repo")
        self.scaler = StandardScaler()
        self.feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "cci",
        ]

        # index to indicate which column is target
        self.target_index = self.feature_columns.index("close")

        self.window_size = config["data_params"]["window_size"]
        self.forecast_steps = config["data_params"]["forecast_steps"]

        self.batch_size = config["model_params"]["lstm_model"]["batch_size"]
        self.input_size = config["model_params"]["lstm_model"]["input_size"]
        self.hidden_size = config["model_params"]["lstm_model"]["hidden_size"]
        self.num_layers = config["model_params"]["lstm_model"]["num_layers"]
        self.output_size = config["data_params"]["forecast_steps"]
        self.num_epochs = config["model_params"]["lstm_model"]["num_epochs"]
        self.learning_rate = config["model_params"]["lstm_model"]["learning_rate"]

        # path where the fitted scaler will be persisted
        os.makedirs("artifacts", exist_ok=True)
        self.scaler_path = "artifacts/scaler.pkl"

        # initiate model and trainer
        self.model = LSTMModel(
            self.input_size, self.hidden_size, self.num_layers, self.output_size
        )

        self.trainer = Trainer(self.model)

    def get_historical_features_df(self):
        """
        Queries feature store for feature and returns pandas DataFrame with historical features

        Returns:
            pd.DataFrame: DataFrame containing historical features
        """

        try:
            entity_df = pd.read_parquet("amzn_stock_repo/data/entity_df.parquet")

            return self.fs.get_historical_features(
                entity_df=entity_df,
                features=[
                    "amazon_stock_fv:high",
                    "amazon_stock_fv:low",
                    "amazon_stock_fv:open",
                    "amazon_stock_fv:volume",
                    "amazon_stock_fv:rsi",
                    "amazon_stock_fv:cci",
                    "amazon_stock_fv:close",
                ],
            ).to_df()

        except FileNotFoundError as e:
            logger.error(f"Couldn't find entity parquet file : {e}")
            raise

        except Exception as e:
            logger.error(f"Error fetching historical features: {e}")
            raise

    def split_into_train_val_and_test(
        self, training_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits training data into train, val and test splits(70, 15, 15)

        Args:
            training_df (pd.DataFrame): DataFrame containing historical features

        Returns:
            tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
            train, val and test dataframes
        """

        try:
            # sort the data by datetime column
            training_df = training_df.sort_values("datetime").reset_index(drop=True)
            # reorder the columns
            training_df = training_df[self.feature_columns]

            # split into 80% train and 20% test
            n = len(training_df)

            train_end = int(n * 0.7)
            val_end = int(n * 0.85)

            train_df = training_df.iloc[:train_end]
            val_df = training_df.iloc[train_end:val_end]
            test_df = training_df.iloc[val_end:]

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"Failed to create train, val and test sets: {e}")
            raise

    def scale_and_generate_sequences(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Scale the train, validation, and test datasets, then generate
        input and target sequences for time series forecasting.

        The scaler is fit only on the training data, then reused to
        transform validation and test data to avoid data leakage. The
        fitted scaler is persisted to disk so it can be reused later
        for inference (e.g. inverse-transforming predictions).

        Args:
            train (pd.DataFrame): Training data of shape.
            val (pd.DataFrame): Validation data of shape.
            test (pd.DataFrame): Test data of shape.

        Returns:
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]:
                A tuple containing:

                - x_train: Training input sequences of shape
                `(num_train_sequences, window_size, num_features)`.
                - y_train: Training target sequences of shape
                `(num_train_sequences, forecast_steps)`.
                - x_val: Validation input sequences of shape
                `(num_val_sequences, window_size, num_features)`.
                - y_val: Validation target sequences of shape
                `(num_val_sequences, forecast_steps)`.
                - x_test: Test input sequences of shape
                `(num_test_sequences, window_size, num_features)`.
                - y_test: Test target sequences of shape
                `(num_test_sequences, forecast_steps)`.
        """

        try:
            # normalizing the data
            train_transformed = self.scaler.fit_transform(train)
            val_transformed = self.scaler.transform(val)
            test_transformed = self.scaler.transform(test)

            # persist the fitted scaler so it can be reused at inference time
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Saved fitted scaler to {self.scaler_path}")

            # generating sequential data
            x_train, y_train = generate_sequences(
                train_transformed,
                self.target_index,
                self.window_size,
                self.forecast_steps,
            )
            x_val, y_val = generate_sequences(
                val_transformed,
                self.target_index,
                self.window_size,
                self.forecast_steps,
            )
            x_test, y_test = generate_sequences(
                test_transformed,
                self.target_index,
                self.window_size,
                self.forecast_steps,
            )

            return x_train, y_train, x_val, y_val, x_test, y_test

        except Exception as e:
            logger.error(f"Couldn't generate sequences: {e}")
            raise

    def prepare_data(self):
        """
        Prepare train, validation, and test DataLoaders for model training.

        Orchestrates the full data pipeline: loading historical features,
        splitting into train/val/test, scaling and generating sequences,
        and wrapping everything into PyTorch DataLoaders.

        Note:
            Each step (loading, splitting, sequence generation, dataset/
            dataloader creation) handles and logs its own exceptions.
            This method does not add additional exception handling to
            avoid redundant catch/log/raise chains.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]:
                (train_loader, val_loader, test_loader)
        """
        # load the data
        logger.info("loading historical features ...")
        training_df = self.get_historical_features_df()

        # split into train, val and test
        logger.info("splitting into train, val and test splits")
        train, val, test = self.split_into_train_val_and_test(training_df)

        # generate sequences
        logger.info("converting to sequences ...")
        x_train, y_train, x_val, y_val, x_test, y_test = (
            self.scale_and_generate_sequences(train, val, test)
        )

        # convert to torch dataset
        logger.info("converting to dataloaders ... ")
        train_set = AmazonStockDataset(x_train, y_train)
        val_set = AmazonStockDataset(x_val, y_val)
        test_set = AmazonStockDataset(x_test, y_test)

        # create loaders
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        return train_loader, val_loader, test_loader

    def train_and_evaluate_model(self, train_loader, val_loader, test_loader):
        """
        Train the model via MLflow-tracked training, evaluate it on the
        held-out test set, and register the model if it outperforms the
        current champion.

        Returns:
            nn.Module: The trained model.
        """
        # build the params dict fit_with_mlflow expects
        params = {
            "window_size": self.window_size,
            "forecast_steps": self.forecast_steps,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
        }

        # train the model — returns both the trained model and the run_id
        # it was logged under, since the run itself closes before this returns
        trained_model, run_id = self.trainer.fit_with_mlflow(
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            params=params,
        )

        # evaluate on the untouched test set
        test_rmse = self.trainer.evaluate(trained_model, test_loader)
        logger.info(f"Test RMSE: {test_rmse:.4f}")

        # log the test metric against the same run (run has already closed,
        # so we need to reopen it by run_id rather than starting a new one)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_rmse", test_rmse)

        # register this run's model and promote to champion if it's better
        self._register_and_maybe_promote(run_id, test_rmse)

        return trained_model

    def _register_and_maybe_promote(self, run_id: str, test_rmse: float):
        """
        Register the model from this run into the MLflow Model Registry,
        and promote it to the "champion" alias if it beats the current
        champion's test RMSE. If no champion exists yet, this run becomes
        the champion by default.

        Args:
            run_id (str): The run ID under which the trained model was logged.
            test_rmse (float): This run's RMSE on the held-out test set.
        """
        client = mlflow.MlflowClient()
        model_name = "amazon-stock-lstm"

        # register a new version of the model from this run's artifacts
        result = mlflow.register_model(
            f"runs:/{run_id}/amazon_stock_price_prediction_model_lstm",
            model_name,
        )
        logger.info(f"Registered {model_name} version {result.version}")

        # fetch the current champion's test RMSE, if a champion exists
        try:
            champion = client.get_model_version_by_alias(model_name, "champion")
            champion_rmse = client.get_run(champion.run_id).data.metrics.get( # pyright: ignore[reportArgumentType]
                "test_rmse", float("inf")
            )  # type:ignore
        except exceptions.MlflowException:
            # no champion registered yet — this run wins by default
            champion_rmse = float("inf")

        # promote only if this run's test RMSE is better (lower)
        if test_rmse < champion_rmse:  # type:ignore
            client.set_registered_model_alias(model_name, "champion", result.version)
            logger.info(
                f"New champion: version {result.version} (test RMSE {test_rmse:.4f})"
            )
        else:
            logger.info(
                f"Kept existing champion (RMSE {champion_rmse:.4f} vs new {test_rmse:.4f})"  # type:ignore
            )

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline...")

        pipeline = TrainingPipeline()

        train_loader, val_loader, test_loader = pipeline.prepare_data()

        model = pipeline.train_and_evaluate_model(train_loader, val_loader, test_loader)

        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise