from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import load_test_dataset
from models.infer import load_model


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    test = load_test_dataset(cfg)
    test["answer"] = 0
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)
    results = load_model(cfg, cfg.models.results)
    folds = np.unique((test["day"] // cfg.data.n_splits) + 1)

    for num in tqdm(test["building_number"].unique()):
        test_x = test[test["building_number"] == num].reset_index(drop=True)
        test_x = test_x.drop(columns=["building_number", "answer"])
        models = results.models

        for fold in folds:
            model = models[f"building_{num}-fold_{fold}"]
            pred = model.predict(test_x)
            test.loc[test["building_number"] == num, "answer"] += pred / len(folds)

    submit["answer"] = test["answer"].to_numpy()

    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
