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
    model_path = Path(get_original_cwd()) / cfg.models.path / cfg.models.name

    if cfg.models.name == "lightgbm":
        model_half_1 = lgb.Booster(model_file=model_path / f"{cfg.models.results}_half_1.txt")
        model_half_2 = lgb.Booster(model_file=model_path / f"{cfg.models.results}_half_2.txt")
        pred = model_half_1.predict(test_x) / 2
        pred += model_half_2.predict(test_x) / 2

        submit["answer"] = pred
        submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
