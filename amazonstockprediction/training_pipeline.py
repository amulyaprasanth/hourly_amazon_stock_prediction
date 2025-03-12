import os
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from torch import nn
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
from sklearn.metrics import root_mean_squared_error
from amazonstockprediction.utils import (
    get_feature_store,
    get_train_test_val_dates,
    generate_sequence,
    read_yaml,
    fit
)
from amazonstockprediction.logger import setup_logger

# Setup logger
logger = setup_logger("training_pipeline")

# Get the config
config = read_yaml()

def get_or_create_feature_view(
    feature_store: FeatureStore,
) -> Tuple[FeatureView, str]:
    """Create a feature view in Hopsworks from the selected features in the provided feature group."""
    try:
        amazon_fg = feature_store.get_feature_group(config["data_params"]["feature_group_name"], version=1)
        selected_features = amazon_fg.select(
            ["datetime", "open", "high", "close", "low", "volume", "rsi", "cci"]
        )

        start_date = pd.DataFrame(amazon_fg.show(1))["id"][0].split()[0]

        amazon_fv = feature_store.get_or_create_feature_view(
            name=config["data_params"]["feature_view_name"],
            version=1,
            query=selected_features,
        )
        return amazon_fv, start_date
    except Exception as e:
        logger.error(f"Error creating feature view: {e}")
        raise

def get_time_series_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    window_size: int = config['data_params']['window_size'],
    forecast_steps: int = config['data_params']['forecast_steps'],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into train, val and test sets and generate sequences."""
    try:


        # Sort the values by date and remove the datetime column
        train = train.sort_values("datetime").drop("datetime", axis=1)
        val = val.sort_values("datetime").drop("datetime", axis=1)
        test = test.sort_values("datetime").drop("datetime", axis=1)

        X_train, y_train = generate_sequence(
            train, window_size=window_size, forecast_steps=forecast_steps
        )
        X_val, y_val = generate_sequence(
            val, window_size=window_size, forecast_steps=forecast_steps
        )
        X_test, y_test = generate_sequence(
            test, window_size=window_size, forecast_steps=forecast_steps
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        logger.error(f"Error generating time series data: {e}")
        raise


class AmazonStockDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def train_and_evaluate_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[nn.Module, float]:
    try:
        # Create dataloaders
        logger.info("Creating dataset and dataloaders...")
        train_dataset = AmazonStockDataset(X_train, y_train)
        val_dataset = AmazonStockDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        # Create model
        logger.info("Creating LSTM model...")
        model = LSTMModel(
           input_size = config["model_params"]["lstm_model"]["input_size"],
           hidden_size = config["model_params"]["lstm_model"]["hidden_size"],
           num_layers = config["model_params"]["lstm_model"]["num_layers"],
           output_size= config["data_params"]["forecast_steps"]
        )

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = config["model_params"]["lstm_model"]["num_epochs"]

        # Train the model 
        logger.info("Training...")
        fit(model, train_loader, val_loader, loss_fn, optimizer, logger, num_epochs)
       
        logger.info("Model Evaluation...")
        model.eval()
        with torch.no_grad():
            y_pred_val = model(torch.tensor(X_val, dtype=torch.float32))

        val_rmse =float(root_mean_squared_error(y_val, y_pred_val))

        logger.info(f"Validation RMSE: {val_rmse}")

        return model, val_rmse
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise

def save_model(model: nn.Module, model_dir: str, model_filename: str):
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model, model_filename)
        logger.info(f"Model saved to {model_filename}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    try:
        fs, mr = get_feature_store()
        amazon_fv, start_date = get_or_create_feature_view(fs)

        (
            train_start_dt,
            train_end_dt,
            val_start_dt,
            val_end_dt,
            test_start_dt,
            test_end_dt,
        ) = get_train_test_val_dates(start_date, train_size=config["data_params"]["train_size"], val_size=config["data_params"]["val_size"])

        train, val, test, _, _, _ = amazon_fv.train_validation_test_split(
            train_start=train_start_dt,
            train_end=train_end_dt,
            val_start=val_start_dt,
            val_end=val_end_dt,
            test_start=test_start_dt,
            test_end=test_end_dt,
        )

        X_train, y_train, X_val, y_val, X_test, y_test = get_time_series_data(
            train, val, test)

        model, val_rmse = train_and_evaluate_model(X_train, y_train, X_val, y_val)

        model_dir = config["model_params"]['lstm_model']["model_dir"]
        model_filepath = os.path.join(model_dir, config['model_params']['lstm_model']['model_filename'])
        save_model(model, model_dir, model_filepath)

        metrics = {"rmse": val_rmse}

        model = mr.torch.create_model(
            name=config["model_params"]['lstm_model']["model_name"],
            description="LSTM Torch model for predicting Amazon stock prices",
            input_example=X_train[0],
            feature_view=amazon_fv,
            metrics = metrics
        )   
        model.save(model_dir)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
