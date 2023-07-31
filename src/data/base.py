import pickle
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BaseDataset:
    def __init__(self, config: DictConfig):
        self.config = config

    def categorize_train_features(self, train: pd.DataFrame) -> pd.DataFrame:
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

        for cat_feature in tqdm(self.config.features.categorical_features, leave=False):
            train[cat_feature] = le.fit_transform(train[cat_feature])
            with open(path / f"{cat_feature}.pkl", "wb") as f:
                pickle.dump(le, f)

        return train

    def categorize_test_features(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.config.data.encoder

        for cat_feature in tqdm(self.config.features.categorical_features, leave=False):
            le = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
            test[cat_feature] = le.transform(test[cat_feature])

        return test


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int):
        self.df = df
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.df[idx : idx + self.window_size, :], dtype=torch.float)
        if self.df.shape[1] > 1:
            y = torch.tensor(self.df[idx + self.window_size, -1], dtype=torch.float)
        else:
            y = None
        return x, y


def create_data_loader(df: pd.DataFrame, window_size: int, batch_size: int) -> torch.Tensor:
    dataset = TimeSeriesDataset(df, window_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader
