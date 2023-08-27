import pickle
from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class BaseDataPreprocessor:
    def __init__(self, config: DictConfig):
        self.config = config

    def _categorize_train_features(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.config.data.encoder
        le = LabelEncoder()

        for cat_feature in tqdm(self.config.data.categorical_features, leave=False):
            train[cat_feature] = le.fit_transform(train[cat_feature])
            with open(path / f"{cat_feature}.pkl", "wb") as f:
                pickle.dump(le, f)

        return train

    def _categorize_test_features(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.config.data.encoder

        for cat_feature in tqdm(self.config.data.categorical_features, leave=False):
            with open(path / f"{cat_feature}.pkl", "rb") as f:
                le = pickle.load(f)
            test[cat_feature] = le.transform(test[cat_feature])

        return test


def categorize_tabnet_features(cfg: DictConfig, train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """
    categorical_columns = []
    categorical_dims = {}

    label_encoder = LabelEncoder()

    for cat_feature in tqdm(cfg.features.categorical_features):
        train[cat_feature] = label_encoder.fit_transform(train[cat_feature].values)
        test[cat_feature] = label_encoder.transform(test[cat_feature].values)
        categorical_columns.append(cat_feature)
        categorical_dims[cat_feature] = len(label_encoder.classes_)

    features = [col for col in train.columns if col not in [cfg.data.target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    return cat_idxs, cat_dims
