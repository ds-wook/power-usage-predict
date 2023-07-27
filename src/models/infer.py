import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from models.base import ModelResult


def load_model(config: DictConfig, model_name: str) -> ModelResult:
    """
    Load model
    Args:
        model_name: model name
    Returns:
        ModelResult object
    """
    model_path = Path(get_original_cwd()) / config.models.path / config.models.name / model_name

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result


def inference(result: ModelResult, test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(result.models)
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(result.models.values(), total=folds):
        preds += (
            model.predict(test_x) / folds
            if isinstance(model, lgb.Booster)
            else model.predict(xgb.DMatrix(test_x)) / folds
            if isinstance(model, xgb.Booster)
            else model.predict(test_x)[:, 1] / folds
        )

    assert len(preds) == len(test_x)

    return preds
