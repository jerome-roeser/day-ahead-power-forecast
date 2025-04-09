import torch
import torch.nn as nn
import torch.nn.functional as F


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


def compute_regression_metrics(model, dataloader):
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
