from typing import Tuple

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BaselinePV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[:, 12:-12, 4]
        return x.view(len(x), -1, 1)


class BaselineForecast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[:, :, 0]
        return x.view(len(x), -1, 1)


class RNNModel(nn.Module):
    def __init__(self, p: int):
        super(RNNModel, self).__init__()
        self.n_features = p
        self.lstm = nn.RNN(p, 24, batch_first=True)
        self.linear = nn.Linear(24, 24)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.tanh(x)
        x = self.linear(x[:, -1, :])
        return x.view(len(x), -1, 1)


class LSTMModel(nn.Module):
    def __init__(self, p: int):
        super(LSTMModel, self).__init__()
        self.n_features = p
        self.lstm = nn.LSTM(p, 24, batch_first=True)
        self.linear = nn.Linear(24, 24)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.tanh(x)
        x = self.linear(x[:, -1, :])
        return x.view(len(x), -1, 1)


class LSTMModel_2(nn.Module):
    def __init__(self, p: int):
        super(LSTMModel_2, self).__init__()
        self.n_features = p
        self.lstm1 = nn.LSTM(p, 24, batch_first=True)
        self.lstm2 = nn.LSTM(24, 24, batch_first=True)
        self.linear1 = nn.Linear(24, 16)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(16, 24)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = F.tanh(x)
        x, _ = self.lstm2(x)
        x = F.tanh(x)
        x = self.linear1(x[:, -1, :])
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x.view(len(x), -1, 1)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def training_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> Tuple[float, float, float, float]:
    """
    Train the model for one epoch and evaluate on the validation set.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_dataloader : DataLoader
        The DataLoader for the training set.
    val_dataloader : DataLoader
        The DataLoader for the validation set.
    loss_fn : nn.Module
        The loss function to use.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    batch_size : int
        The batch size used for training.

    Returns
    -------
    Tuple[float, float, float, float]
        The training loss, validation loss, training MAE, and validation MAE.
    """

    size = len(train_dataloader.dataset)
    running_loss = 0
    for batch, data in enumerate(train_dataloader):
        X, y = data
        output = model(X)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        mae = torch.mean(abs(output - y))

        if batch % 10 == 9:
            loss, current = loss.item(), batch * batch_size + len(X)
            mlflow.log_metric("train_loss", loss, step=current)
            mlflow.log_metric("train_mae", mae, step=current)
            print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    with torch.no_grad():
        outputs = []
        labels = []
        vsize = len(val_dataloader.dataset)
        running_vloss = 0.0

        # In evaluation mode some model specific operations can be omitted
        #  -> eg. dropout layer
        # Switching to evaluation mode, eg. turning off regularisation
        model.eval()
        for j, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            outputs.append(voutputs)
            labels.append(vlabels)
            vloss = loss_fn(voutputs, vlabels)
            vmae = torch.mean(abs(voutputs - vlabels))
            running_vloss += vloss.item()

            if j % 10 == 9:
                vloss, vcurrent = vloss.item(), j * batch_size + len(vinputs)
                mlflow.log_metric("val_loss", vloss, step=vcurrent)
                mlflow.log_metric("val_mae", vmae, step=vcurrent)
                print(f"\tval loss: {vloss:>7f}  [{vcurrent:>5d}/{vsize:>5d}]")

        model.train(True)

    return loss, vloss, mae, vmae


def compute_regression_metrics(model: nn.Module, dataloader: DataLoader) -> dict:
    """
    Compute regression metrics for a given model and dataloader.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataloader : DataLoader
        The dataloader containing the dataset.

    Returns
    -------
    dict: dict
        A dictionary containing the computed metrics: MSE, RMSE, MAE, and R2.
    """
    y_preds = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            y_preds.append(outputs.cpu())
            labels.append(targets.cpu())

    y_preds = torch.cat(y_preds)
    labels = torch.cat(labels)

    # Compute metrics
    mse = torch.mean((y_preds - labels) ** 2)
    rmse = mse**0.5
    mae = torch.mean(abs(y_preds - labels))
    r2 = 1 - torch.sum((y_preds - labels) ** 2) / torch.sum(
        y_preds - torch.mean(labels) ** 2
    )

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
