import pickle
from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from features.engine import FeatureEngineering


def load_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    train = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / "train.csv")
    building_info = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.building_info)
    train = train.rename(columns={**cfg.data.dataset_rename})
    train = train.drop(columns=[*cfg.features.drop_train_features])
    building_info = building_info.rename(columns={**cfg.data.building_info_rename})
    translation_dict = {**cfg.data.building_type_translation}

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    train = pd.merge(train, building_info, on="building_number", how="left")

    # feature engineering
    feature_engineering = FeatureEngineering(config=cfg, df=train)
    train = feature_engineering.get_train_pipeline()

    with open(Path(get_original_cwd()) / cfg.data.encoder / "cluster_map.pkl", "rb") as f:
        cluster_map = pickle.load(f)

    train["cluster"] = train["building_number"].map(cluster_map)

    train = train.drop(columns=[*cfg.features.drop_features])

    return train


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / "test.csv")
    building_info = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.building_info)
    test = test.rename(columns={**cfg.data.dataset_rename})

    building_info = building_info.rename(columns={**cfg.data.building_info_rename})
    translation_dict = {**cfg.data.building_type_translation}

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    test = pd.merge(test, building_info, on="building_number", how="left")

    # add feature
    feature_engineering = FeatureEngineering(config=cfg, df=test)
    test = feature_engineering.get_test_pipeline()

    with open(Path(get_original_cwd()) / cfg.data.encoder / "cluster_map.pkl", "rb") as f:
        cluster_map = pickle.load(f)

    test["cluster"] = test["building_number"].map(cluster_map)

    test_x = test.drop(columns=[*cfg.features.drop_features])

    return test_x
