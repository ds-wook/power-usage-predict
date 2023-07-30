from __future__ import annotations

import gc
import logging
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
from omegaconf import DictConfig

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

    def save_model(self, model_path: Path) -> NoReturn:
        """
        Save model
        """
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

    def train_cross_validation(self, train: pd.DataFrame) -> BaseModel:
        models = dict()
        oof_preds = train[["building_number", self.config.data.target]].copy()
        oof_preds = oof_preds.rename(columns={self.config.data.target: "oof_preds"})
        y_label = train[self.config.data.target]

        for num in train["building_number"].unique():
            train_x = train[train["building_number"] == num].reset_index(drop=True)
            train_y = train_x[self.config.data.target]
            train_x = train_x.drop(columns=["building_number", self.config.data.target])
            train_x["fold_num"] = train_x["hour"] % 12
            oof_pred = np.zeros(len(train_x))

            for fold, idx in enumerate(train_x["fold_num"].unique(), 1):
                train_idx = train_x[train_x["fold_num"] != idx].index
                valid_idx = train_x[train_x["fold_num"] == idx].index
                x_train = train_x.drop(columns=["fold_num"])

                X_train, y_train = x_train.loc[train_idx], train_y.loc[train_idx]
                X_valid, y_valid = x_train.loc[valid_idx], train_y.loc[valid_idx]

                wandb.init(
                    entity=self.config.log.entity,
                    project=self.config.log.project,
                    name=f"building_number-{num}-fold-{fold}",
                )

                model = self._fit(X_train, y_train, X_valid, y_valid)

                oof_pred[valid_idx] = (
                    model.predict(X_valid)
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(X_valid))
                    if isinstance(model, xgb.Booster)
                    else model.predict(X_valid)
                )

                models[f"building_{num}-fold_{fold}"] = model

                del X_train, y_train, X_valid, y_valid, model
                gc.collect()

                wandb.finish()

            oof_preds.loc[train["building_number"] == num, "oof_preds"] = oof_pred

        self.oof_preds = oof_preds
        self.result = ModelResult(oof_preds=oof_preds, models=models)
        logging.info(f"oof score: {smape(y_label, oof_preds['oof_preds'].to_numpy())}")
        print(smape(y_label, oof_preds["oof_preds"].to_numpy()))

        return self
