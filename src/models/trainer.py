import logging
from math import sqrt

import torch
from torch import nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self, model, learning_rate: float = 1e-3, device: torch.device | None = None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def fit_with_mlflow(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        logger: logging.Logger,
        params: dict,
    ) -> nn.Module:
        """
        Train the model, logging parameters, metrics, and the final model to MLflow.
        """
        try:
            for epoch in tqdm(range(params["num_epochs"])):
                train_loss, val_loss = 0.0, 0.0
                self.model.train()

                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X).squeeze(-1)
                    loss = self.loss_fn(outputs, y)
                    train_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()

                with torch.inference_mode():
                    for X, y in val_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        outputs = self.model(X).squeeze(-1)
                        loss = self.loss_fn(outputs, y)
                        val_loss += loss.item()

                train_loss /= len(train_loader)
                val_loss /= len(val_loader)

                logger.info(
                    f"Epoch [{epoch + 1}/{params['num_epochs']}], "
                    f"Train RMSE: {sqrt(train_loss):.4f}, Val RMSE: {sqrt(val_loss):.4f}"
                )


            return self.model

        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def evaluate(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
        """Compute RMSE of `model` on `loader`. Used for test-set evaluation after training."""
        model = model.to(self.device)
        model.eval()
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        with torch.inference_mode():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X).squeeze(-1)
                total_loss += loss_fn(outputs, y).item()
        return sqrt(total_loss / len(loader))
