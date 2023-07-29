from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from omegaconf import DictConfig

from evaluation.metrics import smape
from models.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.models.params),
            num_boost_round=self.config.models.num_boost_round,
            feval=self._evaluation,
            callbacks=[
                wandb_lgb.wandb_callback(),
                lgb.log_evaluation(self.config.models.verbose_eval),
                lgb.early_stopping(self.config.models.early_stopping_rounds),
            ],
        )

        wandb_lgb.log_summary(model)

        return model

    def _evaluation(self, preds: pd.Series | np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
        """
        Custom Evaluation Function for LGBM
        """
        labels = train_data.get_label()
        smape_val = smape(preds, labels)

        return "SMAPE", smape_val, False


class CatBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> CatBoostRegressor:
        train_set = Pool(X_train, y_train, cat_features=self.config.features.categorical_features)
        valid_set = Pool(X_valid, y_valid, cat_features=self.config.features.categorical_features)

        model = CatBoostRegressor(
            random_state=self.config.models.seed,
            **self.config.models.params,
            cat_features=self.config.features.categorical_features,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.config.models.verbose_eval,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            callbacks=wandb_cb.WandbCallback() if self.config.models.params.task_type == "CPU" else None,
        )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            dict(self.config.models.params),
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose_eval=self.config.models.verbose_eval,
            callbacks=[wandb_xgb.WandbCallback()] if self.config.log.experiment else None,
        )

        return model
