from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from models.boosting import LightGBMTrainer, CatBoostTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    X_train, y_train, X_valid, y_valid = load_train_dataset(cfg)
    save_path = Path(get_original_cwd()) / cfg.models.path / cfg.models.name
    if cfg.models.name == "lightgbm":
        # train model
        lgb_trainer = LightGBMTrainer(config=cfg)
        model = lgb_trainer.train(X_train, y_train, X_valid, y_valid)

        # save model
        model.save_model(save_path / cfg.models.results, num_iteration=model.best_iteration)

    elif cfg.models.name == "catboost":
        cb_trainer = CatBoostTrainer(config=cfg)
        model = cb_trainer.train(X_train, y_train, X_valid, y_valid)

        # save model
        model.save_model(save_path / cfg.models.results)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
