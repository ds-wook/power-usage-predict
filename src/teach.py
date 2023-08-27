from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer

from data.dataset import load_test_dataset, load_train_dataset
from evaluation.metrics import SMAPE, smape
from features.base import categorize_tabnet_features
from models.infer import load_model
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/", config_name="teach")
def _main(cfg: DictConfig):
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)
    train = load_train_dataset(cfg)
    train = train.fillna(-999)
    train = reduce_mem_usage(train)
    test_x = load_test_dataset(cfg)
    test_x = test_x.fillna(-999)
    test_x = reduce_mem_usage(test_x)
    train_x = train.drop(columns=[cfg.data.target])
    train_y = train[cfg.data.target]

    for i, (oof, pred) in enumerate(zip(cfg.oofs, cfg.preds)):
        oof_preds = load_model(cfg, oof)
        preds = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / pred)
        train_x[f"oof_preds_{i}"] = oof_preds.oof_preds["oof_preds"].to_numpy()
        test_x[f"oof_preds_{i}"] = preds["answer"].to_numpy()

    scaler = QuantileTransformer(n_quantiles=100, random_state=42, output_distribution="normal")
    columns = [
        c
        for c in train_x.columns
        if c not in [*cfg.features.categorical_features] + ["building_number"] and "oof_preds" not in c
    ]

    train_x[columns] = scaler.fit_transform(train_x[columns])
    test_x[columns] = scaler.transform(test_x[columns])

    cat_idx, cat_dims = categorize_tabnet_features(cfg, train_x, test_x)

    kf = StratifiedGroupKFold(n_splits=cfg.data.n_split, shuffle=False)
    train_x["fold_num"] = train_x["day"] // cfg.data.n_splits
    folds = kf.split(train_x, train_x["fold_num"], groups=train_x["day"])
    train_x = train_x.drop(columns=["fold_num", "day", cfg.data.target])

    oof_preds = np.zeros((train.shape[0], 1))
    blending_preds = np.zeros(test_x.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(folds, 1):
        print(f"Train fold: {fold}")
        X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
        X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

        model = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=cfg.models.params.lr),
            scheduler_params={
                "step_size": cfg.models.params.step_size,
                "gamma": cfg.models.params.gamma,
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=cfg.models.params.mask_type,
            n_steps=cfg.models.params.n_steps,
            n_d=cfg.models.params.n_d,
            n_a=cfg.models.params.n_a,
            lambda_sparse=cfg.models.params.lambda_sparse,
            verbose=cfg.models.params.verbose,
        )
        model.fit(
            X_train=X_train.to_numpy(),
            y_train=y_train.to_numpy().reshape(-1, 1),
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)),
                (X_valid.to_numpy(), y_valid.to_numpy().reshape(-1, 1)),
            ],
            eval_name=[*cfg.models.eval_name],
            eval_metric=[*cfg.models.eval_metric, SMAPE],
            max_epochs=cfg.models.params.max_epochs,
            patience=cfg.models.params.patience,
            batch_size=cfg.models.params.batch_size,
            virtual_batch_size=cfg.models.params.virtual_batch_size,
            num_workers=cfg.models.params.num_workers,
            drop_last=False,
        )

        oof_preds[valid_idx] = model.predict(X_valid.to_numpy())
        blending_preds += model.predict(test_x.to_numpy()).flatten() / kf.n_splits

        del X_train, y_train, X_valid, y_valid, model

    print(f"Stacking Score: {smape(oof_preds.flatten(), train_y)}")
    submit["answer"] = blending_preds
    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
