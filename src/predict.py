from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_test_dataset


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    test_x = load_test_dataset(cfg)
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)

    if cfg.models.name == "lightgbm":
        model = lgb.Booster(model_file=Path(get_original_cwd()) / cfg.models.path / cfg.models.results)
        pred = model.predict(test_x)
        submit["answer"] = pred
        submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
