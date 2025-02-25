import torch
import torch.nn as nn
import torch.nn.functional as F


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
    rmse = mse ** 0.5
    mae = torch.mean(abs(y_preds - labels))
    r2 = 1 - torch.sum((y_preds - labels) ** 2) / torch.sum(y_preds - torch.mean(labels) ** 2)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


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
        self.lstm = nn.RNN(p, 24, batch_first= True)
        self.linear = nn.Linear(24,24)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.tanh(x)
        x = self.linear(x[:,-1,:])
        return x.view(len(x), -1, 1)

class LSTMModel(nn.Module):
    def __init__(self, p: int):
        super(LSTMModel, self).__init__()
        self.n_features = p
        self.lstm = nn.LSTM(p, 24, batch_first= True)
        self.linear = nn.Linear(24,24)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.tanh(x)
        x = self.linear(x[:,-1,:])
        return x.view(len(x), -1, 1)
