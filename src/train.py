from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import wandb
import wandb.lightgbm as wandb_lgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from models.boosting import lgbm_smape


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    X_train, y_train, X_valid, y_valid = load_train_dataset(cfg)

    if cfg.models.name == "lightgbm":
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = lgb.Dataset(X_valid, y_valid)

        # wandb.init(entity=cfg.log.entity, project=cfg.log.project, name=cfg.log.name)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(cfg.models.params),
            num_boost_round=cfg.models.num_boost_round,
            feval=lgbm_smape,
            callbacks=[
                # wandb_lgb.wandb_callback(),
                lgb.log_evaluation(cfg.models.verbose_eval),
                lgb.early_stopping(cfg.models.early_stopping_rounds),
            ],
        )

        save_path = Path(get_original_cwd()) / cfg.models.path
        model.save_model(save_path / cfg.models.results, num_iteration=model.best_iteration)
        # wandb.finish()


if __name__ == "__main__":
    _main()
