from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor

from evaluation.metrics import SMAPE
from models.base import BaseModel


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


class TabNetTrainer(BaseModel):
    def __init__(self, config: DictConfig, cat_idxs: list[int] = [], cat_dims: list[int] = []):
        super().__init__(config)
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> TabNetRegressor:
        """method train"""
        model = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.config.models.params.lr),
            scheduler_params={
                "step_size": self.config.models.params.step_size,
                "gamma": self.config.models.params.gamma,
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=self.config.models.params.mask_type,
            n_steps=self.config.models.params.n_steps,
            n_d=self.config.models.params.n_d,
            n_a=self.config.models.params.n_a,
            lambda_sparse=self.config.models.params.lambda_sparse,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            verbose=self.config.models.params.verbose,
        )

        model.fit(
            X_train=X_train.to_numpy(),
            y_train=y_train.to_numpy().reshape(-1, 1),
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)),
                (X_valid.to_numpy(), y_valid.to_numpy().reshape(-1, 1)),
            ],
            eval_name=[*self.config.models.eval_name],
            eval_metric=[*self.config.models.eval_metric, SMAPE],
            max_epochs=self.config.models.params.max_epochs,
            patience=self.config.models.params.patience,
            batch_size=self.config.models.params.batch_size,
            virtual_batch_size=self.config.models.params.virtual_batch_size,
            num_workers=self.config.models.params.num_workers,
            drop_last=False,
        )

        return model
