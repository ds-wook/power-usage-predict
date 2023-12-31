from __future__ import annotations

from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from omegaconf import DictConfig

from evaluation.metrics import smape
from models.base import BaseModel


class XGBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)

        model = xgb.train(
            dict(self.config.models.params),
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose_eval=self.config.models.verbose_eval,
            obj=partial(self._weighted_mse, alpha=self.config.models.alpha)
            if self.config.models.is_custom_loss
            else None,
            feval=self._evaluation,
        )

        return model

    def _weighted_mse(self, preds: np.ndarray, label: xgb.DMatrix, alpha: int = 1) -> tuple[np.ndarray, np.ndarray]:
        label = label.get_label()
        residual = (label - preds).astype("float")
        grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
        hess = np.where(residual > 0, 2 * alpha, 2.0)
        return grad, hess

    def _evaluation(self, preds: pd.Series | np.ndarray, train_data: xgb.DMatrix) -> tuple[str, float]:
        """
        Custom Evaluation Function for LGBM
        """
        labels = train_data.get_label()
        smape_val = smape(preds, labels)

        return "SMAPE", smape_val


class CatBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> CatBoostRegressor:
        train_set = Pool(X_train, y_train, cat_features=self.config.features.categorical_features)
        valid_set = Pool(X_valid, y_valid, cat_features=self.config.features.categorical_features)

        model = CatBoostRegressor(
            cat_features=self.config.features.categorical_features,
            random_state=self.config.models.seed,
            **self.config.models.params,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.config.models.verbose_eval,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
        )

        return model


class LightGBMTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.config.features.categorical_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.config.features.categorical_features)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.models.params),
            num_boost_round=self.config.models.num_boost_round,
            fobj=partial(self._weighted_mse, alpha=self.config.models.alpha)
            if self.config.models.is_custom_loss
            else None,
            feval=self._evaluation,
            callbacks=[
                lgb.log_evaluation(self.config.models.verbose_eval),
                lgb.early_stopping(self.config.models.early_stopping_rounds),
            ],
        )

        return model

    def _evaluation(self, preds: pd.Series | np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
        """
        Custom Evaluation Function for LGBM
        """
        labels = train_data.get_label()
        smape_val = smape(preds, labels)

        return "SMAPE", smape_val, False
