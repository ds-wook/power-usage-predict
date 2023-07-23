import lightgbm as lgb
import numpy as np
import pandas as pd


def smape(preds: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> float:
    """
    Function to calculate SMAPE
    """
    return 100 / len(preds) * np.sum(2 * np.abs(preds - target) / (np.abs(target) + np.abs(preds)))


def lgbm_smape(self, preds: pd.Series | np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
    """
    Custom Evaluation Function for LGBM
    """
    labels = train_data.get_label()
    smape_val = smape(preds, labels)

    return "SMAPE", smape_val, False
