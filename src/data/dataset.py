from pathlib import Path

import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from features.engine import FeatureEngineering


def map_date_index(date: pd.DatetimeIndex, min_date: int) -> int:
    return (date - min_date).days


def load_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    train = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.train)
    building_info = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.building_info)

    train = train.rename(columns={**cfg.data.dataset_rename})
    train = train.drop(columns=[*cfg.features.drop_train_features])
    building_info = building_info.rename(columns={**cfg.data.building_info_rename})
    translation_dict = {**cfg.data.building_type_translation}

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    train = pd.merge(train, building_info, on="building_number", how="left")

    # feature engineering
    train = FeatureEngineering(config=cfg, df=train).get_train_preprocessed()

    if cfg.models.name != "n_beats":
        train = train.drop(columns=[*cfg.features.drop_features])

    else:
        min_date = train["date_time"].min()
        train["time_idx"] = train["date_time"].map(lambda date: map_date_index(date, min_date))
        train = train.drop(columns=[*cfg.features.drop_features])

    return train


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.test)
    building_info = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.building_info)

    test = test.rename(columns={**cfg.data.dataset_rename})

    building_info = building_info.rename(columns={**cfg.data.building_info_rename})
    translation_dict = {**cfg.data.building_type_translation}

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    test = pd.merge(test, building_info, on="building_number", how="left")

    # add feature
    test = FeatureEngineering(config=cfg, df=test).get_test_preprocessed()

    test_x = test.drop(columns=[*cfg.features.drop_features])

    return test_x


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
