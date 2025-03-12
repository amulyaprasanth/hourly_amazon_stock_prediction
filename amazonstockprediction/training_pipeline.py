from codecs import ignore_errors
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from torch import nn
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from amazonstockprediction.utils import (
    get_feature_store,
    get_train_test_dates,
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
    test: pd.DataFrame,
    window_size: int = config['data_params']['window_size'],
    forecast_steps: int = config['data_params']['forecast_steps'],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Scales the data and split the data into train, val and test sets and generate sequences.
    Returns: 
    X_train (np.ndarray): training feature array
    y_train(np.ndarray): training target array
    X_val (np.ndarray): test feature arrray
    y_val (np.ndarray): test target array
    preprocessor (StandardScaler): preprocessor object
    """
    try:


        # Sort the values by date and remove the datetime column and ensure the column order
        train = train.sort_values("datetime").drop("datetime", axis=1)[["open"
                                                                        , "high", "close", "low", "volume", "rsi", "cci"]]
        test = test.sort_values("datetime").drop("datetime", axis=1)[["open"
                                                                        , "high", "close", "low", "volume", "rsi", "cci"]]


        # Apply preprocessor
        preprocessor = StandardScaler()
        train_transformed = preprocessor.fit_transform(train)
        test_transformed = preprocessor.transform(test)
        

        X_train, y_train = generate_sequence(
            train_transformed, window_size=window_size, forecast_steps=forecast_steps
        )
        X_test, y_test = generate_sequence(
            test_transformed, window_size=window_size, forecast_steps=forecast_steps
        )
        return X_train, y_train, X_test, y_test, preprocessor
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
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, preprocessor: StandardScaler
) -> Tuple[nn.Module, float]:
    try:
        # Create dataloaders
        logger.info("Creating dataset and dataloaders...")
        train_dataset = AmazonStockDataset(X_train, y_train)
        test_dataset = AmazonStockDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["model_params"]["lstm_model"]["batch_size"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["model_params"]["lstm_model"]["batch_size"], shuffle=False)

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
        fit(model, train_loader, test_loader, loss_fn, optimizer, logger, num_epochs)
       
        logger.info("Model Evaluation...")
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32))

        # Do the inverse transform to find rmse
        feature_mean = preprocessor.mean_[2] # type: ignore
        feature_std = preprocessor.scale_[2] # type: ignore
        
        # Inverse transform the scaled value
        inverse_transformed_preds = y_pred * feature_std + feature_mean
        test_rmse =float(root_mean_squared_error(y_test, inverse_transformed_preds))

        logger.info(f"Validation RMSE: {test_rmse}")

        return model, test_rmse
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise

def save_model_and_preprocessor(model: nn.Module, preprocessor: StandardScaler, model_dir: str, model_filepath: str, preprocessor_filepath: str):
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        logger.info("Saving model and preprocessor ...")

        torch.save(model, model_filepath)
        joblib.dump(preprocessor, preprocessor_filepath)

        logger.info(f"Model saved to {model_filepath}")
        logger.info(f"Preprocessor saved to {preprocessor_filepath}")
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
            test_start_dt,
            test_end_dt,
        ) = get_train_test_dates(start_date, train_size=config["data_params"]["train_size"])

        train,  test,  _, _ = amazon_fv.train_test_split(
            train_start=train_start_dt,
            train_end=train_end_dt,
            test_start=test_start_dt,
            test_end=test_end_dt,
        )

        X_train, y_train, X_test, y_test, preprocessor= get_time_series_data(
            train, test)

        model, test_rmse = train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor)

        model_dir = config["model_params"]['lstm_model']["model_dir"]
        model_filepath = os.path.join(model_dir, config['model_params']['lstm_model']['model_filename'])
        preprocessor_filepath = os.path.join(model_dir, config['model_params']['preprocessor_filename'])
        save_model_and_preprocessor(model= model,
                                    preprocessor = preprocessor,
                                      model_dir = model_dir, model_filepath=model_filepath, preprocessor_filepath=preprocessor_filepath)

        metrics = {"rmse": test_rmse}

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
