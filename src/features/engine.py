from sklearn.preprocessing import LabelEncoder
from omegaconf import DictConfig
import pandas as pd
import pickle
from pathlib import Path
from hydra.utils import get_original_cwd
from tqdm import tqdm


def categorize_train_features(cfg: DictConfig, train: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / cfg.data.encoder
    le = LabelEncoder()

    for cat_feature in tqdm(cfg.features.categorical_features, leave=False):
        train[cat_feature] = le.fit_transform(train[cat_feature])
        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(le, f)

    return train


def categorize_test_features(cfg: DictConfig, test: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        test: dataframe
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / cfg.data.encoder

    for cat_feature in tqdm(cfg.features.categorical_features, leave=False):
        le = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
        test[cat_feature] = le.transform(test[cat_feature])

    return test
