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
import xgboost as xgb
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import StratifiedGroupKFold

from evaluation.metrics import smape

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig):
        self.config = config

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
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def train_cross_validation(self, train: pd.DataFrame) -> BaseModel:
        models = dict()
        oof_preds = train[["building_number", self.config.data.target]].copy()
        oof_preds = oof_preds.rename(columns={self.config.data.target: "oof_preds"})
        y_label = train[self.config.data.target]
        kf = StratifiedGroupKFold(n_splits=self.config.data.n_splits, shuffle=False)

        for num in train["building_number"].unique():
            train_x = train[train["building_number"] == num].reset_index(drop=True)
            train_y = train_x[self.config.data.target]
            train_x = train_x.drop(columns=["building_number", self.config.data.target])
            train_x["fold_num"] = train_x["day"] // self.config.data.n_splits
            folds = kf.split(train_x, train_x["fold_num"], groups=train_x["day"])
            oof_pred = np.zeros(len(train_x))

            for fold, (train_idx, valid_idx) in enumerate(folds):
                print(f"building: {num} fold: {fold}")
                x_train = train_x.drop(columns=["fold_num", "day"])

                X_train, y_train = x_train.loc[train_idx], train_y.loc[train_idx]
                X_valid, y_valid = x_train.loc[valid_idx], train_y.loc[valid_idx]

                model = self._fit(X_train, y_train, X_valid, y_valid)

                oof_pred[valid_idx] = (
                    model.predict(X_valid)
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(X_valid, enable_categorical=True))
                    if isinstance(model, xgb.Booster)
                    else model.predict(X_valid.to_numpy()).reshape(-1)
                    if isinstance(model, TabNetRegressor)
                    else model.predict(X_valid)
                )

                models[f"building_{num}-fold_{fold}"] = model

                del X_train, y_train, X_valid, y_valid, model
                gc.collect()

            oof_preds.loc[train["building_number"] == num, "oof_preds"] = oof_pred

        self.oof_preds = oof_preds
        self.result = ModelResult(oof_preds=oof_preds, models=models)

        print(f"OOF Score: {smape(oof_preds['oof_preds'], y_label)}\n")

        return self
