import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from feast import FeatureStore
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

        self.registry_path = "src/models/registry.json"

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
        trained_model = self.trainer.fit_with_mlflow(
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            params=params,
        )

        # evaluate on the untouched test set
        test_rmse = self.trainer.evaluate(trained_model, test_loader)
        logger.info(f"Test RMSE: {test_rmse:.4f}")

        return trained_model, test_rmse



    def register_model(self, trained_model: torch.nn.Module, test_rmse: float) -> None:
        """
        Register a trained model if it ranks among the top three models based on RMSE.

        The method performs the following steps:
            1. Loads the existing model registry.
            2. Adds the newly trained model as a candidate.
            3. Sorts all models by RMSE (lower is better).
            4. Retains only the top three models.
            5. Saves the new model only if it is in the top three.
            6. Deletes models that are no longer in the top three.
            7. Updates the registry on disk.

        Args:
            trained_model (torch.nn.Module):
                The trained PyTorch model.

            test_rmse (float):
                RMSE of the model evaluated on the test dataset.

        Raises:
            RuntimeError:
                If the model registration process fails.
        """
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
            model_path = Path(f"src/models/model_{timestamp}.pt")

            # Load registry
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r") as f:
                    registry = json.load(f)
            else:
                registry = []

            candidate = {
                "version": timestamp,
                "rmse": test_rmse,
                "path": str(model_path),
            }

            # Determine the best three models
            candidates = registry + [candidate]
            candidates.sort(key=lambda model: model["rmse"])
            top_models = candidates[:3]

            # Save only if the candidate is in the top three
            if candidate in top_models:
                torch.save(trained_model.state_dict(), model_path)

                # Remove models that are no longer in the top three
                removed_models = [model for model in registry if model not in top_models]

                for model in removed_models:
                    path = Path(model["path"])
                    if path.exists():
                        path.unlink()

                # Update registry
                with open(self.registry_path, "w") as f:
                    json.dump(top_models, f, indent=4)

                logger.info(
                    "Model %s registered successfully with RMSE %.4f.",
                    timestamp,
                    test_rmse,
                )
            else:
                logger.info(
                    "Model RMSE %.4f did not qualify for the top 3. Registration skipped.",
                    test_rmse,
                )

        except (OSError, json.JSONDecodeError, RuntimeError) as e:
            logger.exception("Failed to register model.")
            raise RuntimeError(f"Model registration failed: {e}") from e
        
        
    def run(self):
        """ Run the pipeline """
        try:
            logger.info("Starting training pipeline...")
            
            train_loader, val_loader, test_loader = self.prepare_data()
    
            trained_model, test_rmse = self.train_and_evaluate_model(train_loader, val_loader, test_loader)
            
            self.register_model(trained_model, test_rmse)
    
            logger.info("Training pipeline completed successfully.")
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
