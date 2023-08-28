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
