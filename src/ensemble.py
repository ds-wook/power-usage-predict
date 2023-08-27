from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from tqdm import tqdm

from data.dataset import load_train_dataset
from evaluation.metrics import smape
from models.infer import load_model


def get_score(weights: np.ndarray, train_idx: list[int], oofs: list[np.ndarray], preds: np.ndarray) -> float:
    blending = np.zeros_like(oofs[0][train_idx])

    for oof, weight in zip(oofs[:-1], weights):
        blending += weight * oof[train_idx]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx]

    scores = smape(preds[train_idx], blending)

    return scores


def get_best_weights(oofs: np.ndarray, preds: np.ndarray) -> float:
    weights = np.array([1 / len(oofs) for _ in range(len(oofs) - 1)])
    weight_list = []

    kf = KFold(n_splits=5)
    for fold, (train_idx, _) in enumerate(kf.split(oofs[0]), 1):
        res = minimize(get_score, weights, args=(train_idx, oofs, preds), method="Nelder-Mead", tol=1e-6)
        print(f"fold: {fold} res.x: {res.x}")
        weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    mean_weight = np.insert(mean_weight, len(mean_weight), 1 - np.sum(mean_weight))
    print(f"optimized weight: {mean_weight}\n")

    return mean_weight


@hydra.main(config_path="../config/", config_name="ensemble")
def _main(cfg: DictConfig):
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)
    train = load_train_dataset(cfg)
    target = train[cfg.data.target].to_numpy()

    oofs_list, preds_list = [], []
    for oof_name, pred_name in tqdm(zip(cfg.oofs, cfg.preds[:-1]), total=len(cfg.oofs)):
        oofs = load_model(cfg, oof_name)
        oofs_list.append(oofs.oof_preds["oof_preds"].to_numpy())

        preds = pd.read_csv(Path(get_original_cwd()) / cfg.output.path / pred_name)
        preds_list.append(preds["answer"].to_numpy())

    preds_list.append(pd.read_csv(Path(get_original_cwd()) / cfg.output.path / cfg.preds[-1])["answer"].to_numpy())
    print(f"XGBoost Score: {smape(oofs_list[0], target)}")
    print(f"LightGBM Score: {smape(oofs_list[1], target)}")
    print(f"CatBoost Score: {smape(oofs_list[2], target)}")

    blending_preds = np.median(np.vstack(preds_list), axis=0)

    submit["answer"] = blending_preds
    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
