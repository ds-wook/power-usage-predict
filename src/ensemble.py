from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="ensemble")
def _main(cfg: DictConfig):
    preds_fold1 = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.ensemble.preds1)
    preds_fold2 = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.ensemble.preds2)
    preds_fold3 = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.ensemble.preds3)
    preds_fold4 = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.ensemble.preds4)
    preds_fold5 = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.ensemble.preds5)

    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)

    preds_list = [preds_fold1, preds_fold2, preds_fold3, preds_fold4, preds_fold5]

    preds = [fold["answer"].to_numpy() for fold in preds_list]

    submit["answer"] = np.mean(preds, axis=0)
    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
