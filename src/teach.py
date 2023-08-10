from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from data.dataset import load_test_dataset, load_train_dataset
from evaluation.metrics import SMAPE, smape
from features.base import categorize_tabnet_features
from models.infer import load_model


@hydra.main(config_path="../config/", config_name="teach")
def _main(cfg: DictConfig):
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)
    train = load_train_dataset(cfg)

    train_x = train.drop(columns=[cfg.data.target])
    train_x = train_x.fillna(0)
    oof = load_model(cfg, cfg.oofs[0])
    train_x["oof_preds"] = oof.oof_preds["oof_preds"].to_numpy()
    train_y = train[cfg.data.target]

    preds = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.preds[0])
    test_x = load_test_dataset(cfg)
    test_x = test_x.fillna(0)
    test_x["oof_preds"] = preds["answer"].to_numpy()

    scaler = MinMaxScaler()
    columns = [
        c for c in train_x.columns if c not in [*cfg.features.categorical_features] + ["building_number", "oof_preds"]
    ]
    train_x[columns] = scaler.fit_transform(train_x[columns])
    test_x[columns] = scaler.transform(test_x[columns])
    cat_idx, cat_dims = categorize_tabnet_features(cfg, train_x)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((train.shape[0], 1))
    blending_preds = np.zeros(test_x.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train), 1):
        print(f"Train fold: {fold}")
        X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
        X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

        model = TabNetRegressor(
            cat_idxs=cat_idx,
            cat_dims=cat_dims,
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

    print(f"Stacking Score: {smape(blending_preds, train_y)}")
    submit["answer"] = blending_preds
    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
