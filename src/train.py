from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from models.tree import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(get_original_cwd()) / cfg.models.path

        train = load_train_dataset(cfg)

        if cfg.models.name == "lightgbm":
            # train model
            lgb_trainer = LightGBMTrainer(config=cfg)
            lgb_trainer.train_cross_validation(train)

            # save model
            lgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "catboost":
            # train model
            cb_trainer = CatBoostTrainer(config=cfg)
            cb_trainer.train_cross_validation(train)

            # save model
            cb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "xgboost":
            # train model
            xgb_trainer = XGBoostTrainer(config=cfg)
            xgb_trainer.train_cross_validation(train)

            # save model
            xgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
