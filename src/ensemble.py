from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config/", config_name="ensemble")
def _main(cfg: DictConfig):
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)

    preds_list = []
    for name in tqdm(cfg.ensemble):
        preds = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / name)
        preds_list.append(preds["answer"])

    submit["answer"] = np.mean(preds_list, axis=0)
    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
