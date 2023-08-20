from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from models.infer import load_model


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    test = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.test)
    test["answer"] = 0
    folds = np.unique(np.array(list(range(1, 31 + 1))) // cfg.data.n_splits)
    folds = [i for i in range(folds.shape[0])]
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)

    results = load_model(cfg, f"{cfg.models.results}.pkl")
    for num in tqdm(test["building_number"].unique()):
        test_x = test[test["building_number"] == num].reset_index(drop=True)
        test_x = test_x.drop(columns=["building_number", "answer"])

        for col in cfg.features.categorical_features:
            test_x[col] = test_x[col].astype("category")

        models = results.models

        for fold in tqdm(folds, leave=False):
            model = models[f"building_{num}-fold_{fold}"]
            pred = (
                model.predict(xgb.DMatrix(test_x, enable_categorical=True))
                if isinstance(model, xgb.Booster)
                else model.predict(test_x)
            )
            test.loc[test["building_number"] == num, "answer"] += pred / len(folds)

    submit["answer"] = test["answer"].to_numpy()

    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
