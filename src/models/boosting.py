import numpy as np
import pandas as pd
import lightgbm as lgb


def smape(preds: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> float:
    """
    Function to calculate SMAPE
    """
    return 100 / len(preds) * np.sum(2 * np.abs(preds - target) / (np.abs(target) + np.abs(preds)))


def lgbm_smape(preds: pd.Series | np.ndarray, train_data: lgb.Dataset):
    """
    Custom Evaluation Function for LGBM
    """
    labels = train_data.get_label()
    smape_val = smape(preds, labels)
    return "MAPE", smape_val, False
