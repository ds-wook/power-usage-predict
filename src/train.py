from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler

from data.dataset import load_train_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from models.net import TabNetTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(get_original_cwd()) / cfg.models.path

        if cfg.models.name == "lightgbm":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            lgb_trainer = LightGBMTrainer(config=cfg)
            lgb_trainer.train_cross_validation(train)

            # save model
            lgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "catboost":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            cb_trainer = CatBoostTrainer(config=cfg)
            cb_trainer.train_cross_validation(train)

            # save model
            cb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "xgboost":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            xgb_trainer = XGBoostTrainer(config=cfg)
            xgb_trainer.train_cross_validation(train)

            # save model
            xgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "tabnet":
            # load dataset
            train = load_train_dataset(cfg)
            train = train.fillna(0)
            scaler = MinMaxScaler()
            columns = [
                c
                for c in train.columns
                if c not in [*cfg.features.categorical_features] and c != cfg.data.target and c != "building_number"
            ]
            train[columns] = scaler.fit_transform(train[columns])

            # train model
            tabnet_trainer = TabNetTrainer(config=cfg)
            tabnet_trainer.train_cross_validation(train)

            # save model
            tabnet_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
