import torch
import torch.nn as nn


class EarlyStoppingCallback:
    def __init__(self, min_delta: float = 0.1, patience: int = 5):
        self.min_delta = min_delta
        self.patience = patience
        self.best_epoch_score = 0

        self.attempt = 0
        self.best_score = None
        self.stop_training = False

    def __call__(self, validation_loss: float):
        self.epoch_score = validation_loss

        if self.best_epoch_score == 0:
            self.best_epoch_score = self.epoch_score

        elif self.epoch_score > self.best_epoch_score - self.min_delta:
            self.attempt += 1

            if self.attempt >= self.patience:
                self.stop_training = True

        else:
            self.best_epoch_score = self.epoch_score
            self.attempt = 0


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

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
