from __future__ import annotations

import gc
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import KFold

from evaluation.metrics import smape

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig):
        self.config = config
        self._num_fold_iter = 0
        self.oof_preds = None

    @abstractclassmethod
    def _fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> NoReturn:
        raise NotImplementedError

    def save_model(self):
        """
        Save model
        """
        model_path = (
            Path(get_original_cwd()) / self.config.models.path / self.config.models.name / self.config.models.results
        )

        with open(model_path, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> BaseModel:
        """
        Train model
        """
        wandb.init(entity=self.config.log.entity, project=self.config.log.project, name=self.config.log.name)
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def train_cross_validation(self, train_x: pd.DataFrame, train_y: pd.Series) -> BaseModel:
        models = dict()
        tscv = KFold(n_splits=self.config.data.n_splits)
        oof_preds = np.zeros(train_x.shape[0])

        for fold, (train_idx, valid_idx) in enumerate(tscv.split(train_x), 1):
            x_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            x_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]
            wandb.init(
                entity=self.config.log.entity,
                project=self.config.log.project,
                name=self.config.log.name + f"-fold-{fold}",
            )

            model = self._fit(x_train, y_train, x_valid, y_valid)

            oof_preds[valid_idx] = (
                model.predict(x_valid)
                if isinstance(model, lgb.Booster)
                else model.predict(xgb.DMatrix(x_valid))
                if isinstance(model, xgb.Booster)
                else model.predict(x_valid)
            )

            models[f"fold_{fold}"] = model

            del x_train, y_train, x_valid, y_valid, model
            gc.collect()

            wandb.finish()

        self.oof_preds = oof_preds
        self.result = ModelResult(oof_preds=oof_preds, models=models)
        print(smape(train_y, oof_preds))
        return self
