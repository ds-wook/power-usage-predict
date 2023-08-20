from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import load_test_dataset, load_train_dataset


def make_mean_features(df: pd.DataFrame, power_hour_mean: pd.DataFrame, power_hour_std: pd.DataFrame) -> pd.DataFrame:
    """
    Make mean features
    Args:
        df: dataframe
        power_mean: power mean dataframe
        power_hour_mean: power hour mean dataframe
        power_hour_std: power hour std dataframe
    Returns:
        dataframe
    """
    tqdm.pandas()

    df["hour_mean"] = df.progress_apply(
        lambda x: power_hour_mean.loc[
            (power_hour_mean.building_number == x["building_number"]) & (power_hour_mean.hour == x["hour"]),
            "power_consumption",
        ].values[0],
        axis=1,
    )

    tqdm.pandas()
    df["hour_std"] = df.progress_apply(
        lambda x: power_hour_std.loc[
            (power_hour_std.building_number == x["building_number"]) & (power_hour_std.hour == x["hour"]),
            "power_consumption",
        ].values[0],
        axis=1,
    )

    return df


def make_meta_features(train: pd.DataFrame) -> pd.DataFrame:
    #######################################
    # # 건물별 시간별 발전량 평균 넣어주기
    #######################################
    power_hour_mean = pd.pivot_table(
        train, values="power_consumption", index=["building_number", "hour"], aggfunc=np.median
    ).reset_index()

    #######################################
    # # 건물별 시간별 발전량 표준편차 넣어주기
    #######################################
    power_hour_std = pd.pivot_table(
        train, values="power_consumption", index=["building_number", "hour"], aggfunc=np.std
    ).reset_index()

    return power_hour_mean, power_hour_std


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(get_original_cwd()) / cfg.data.path

        train = load_train_dataset(cfg)
        test = load_test_dataset(cfg)

        power_hour_mean, power_hour_std = make_meta_features(train)

        train = make_mean_features(train, power_hour_mean, power_hour_std)
        test = make_mean_features(test, power_hour_mean, power_hour_std)

        train.to_csv(save_path / "preprocessing_train.csv", index=False)
        test.to_csv(save_path / "preprocessing_test.csv", index=False)


if __name__ == "__main__":
    _main()
