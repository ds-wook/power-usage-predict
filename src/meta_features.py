from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from features.engine import FeatureEngineering


def make_meta_features(train: pd.DataFrame) -> pd.DataFrame:
    #######################################
    # # 건물별, 요일별, 시간별 발전량 평균 넣어주기
    #######################################
    power_mean = pd.pivot_table(
        train, values="power_consumption", index=["building_number", "hour", "day"], aggfunc=np.mean
    ).reset_index()

    #######################################
    # # 건물별 시간별 발전량 평균 넣어주기
    #######################################
    power_hour_mean = pd.pivot_table(
        train, values="power_consumption", index=["building_number", "hour"], aggfunc=np.mean
    ).reset_index()

    #######################################
    # # 건물별 시간별 발전량 표준편차 넣어주기
    #######################################
    power_hour_std = pd.pivot_table(
        train, values="power_consumption", index=["building_number", "hour"], aggfunc=np.std
    ).reset_index()

    return power_mean, power_hour_mean, power_hour_std


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(get_original_cwd()) / cfg.data.path

        train = pd.read_csv(save_path / cfg.data.train)
        building_info = pd.read_csv(save_path / cfg.data.building_info)

        train = train.rename(columns={**cfg.data.dataset_rename})
        train = train.drop(columns=[*cfg.features.drop_train_features])
        building_info = building_info.rename(columns={**cfg.data.building_info_rename})
        translation_dict = {**cfg.data.building_type_translation}

        building_info["building_type"] = building_info["building_type"].replace(translation_dict)

        train = pd.merge(train, building_info, on="building_number", how="left")

        # feature engineering
        feature_engineering = FeatureEngineering(config=cfg, df=train)
        train = feature_engineering.get_train_pipeline()

        # cluster features
        power_mean, power_hour_mean, power_hour_std = make_meta_features(train)

        power_mean.to_csv(save_path / "power_mean.csv", index=False)
        power_hour_mean.to_csv(save_path / "power_hour_mean.csv", index=False)
        power_hour_std.to_csv(save_path / "power_hour_std.csv", index=False)


if __name__ == "__main__":
    _main()
