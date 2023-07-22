import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import get_original_cwd
from features.engine import categorize_train_features, categorize_test_features


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.train)
    building_info = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.building_info)

    train = train.rename(columns={**cfg.data.dataset_rename})
    train = train.drop(columns=[*cfg.features.drop_train_features])
    train["rainfall"] = train["rainfall"].fillna(0.0)
    train["windspeed"] = train["windspeed"].fillna(0.0)
    train["humidity"] = train["humidity"].fillna(0.0)

    building_info = building_info.rename(columns={**cfg.data.building_info_rename})
    translation_dict = {**cfg.data.building_type_translation}

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    train = pd.merge(train, building_info, on="building_number", how="left")
    train["date_time"] = pd.to_datetime(train["date_time"], format="%Y%m%d %H")

    # date time feature 생성
    train["hour"] = train["date_time"].dt.hour
    train["day"] = train["date_time"].dt.day
    train["month"] = train["date_time"].dt.month
    train["weekday"] = train["date_time"].dt.weekday

    train = train.drop(columns=[*cfg.features.drop_features])
    train_x = train.drop(cfg.data.target, axis=1)
    train_x = categorize_train_features(cfg, train_x)
    train_y = train[cfg.data.target]

    X_train = train_x[train_x["date_time"] < cfg.data.split_date]
    X_train = X_train.drop(columns=["date_time"])
    y_train = train_y[train_y.index.isin(X_train.index)]
    X_valid = train_x[train_x["date_time"] >= cfg.data.split_date]
    X_valid = X_valid.drop(columns=["date_time"])
    y_valid = train_y[train_y.index.isin(X_valid.index)]

    return X_train, y_train, X_valid, y_valid


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.test)
    building_info = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.building_info)

    test = test.rename(columns={**cfg.data.dataset_rename})
    test["rainfall"] = test["rainfall"].fillna(0.0)
    test["windspeed"] = test["windspeed"].fillna(0.0)
    test["humidity"] = test["humidity"].fillna(0.0)

    building_info = building_info.rename(columns={**cfg.data.building_info_rename})
    translation_dict = {**cfg.data.building_type_translation}

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    test = pd.merge(test, building_info, on="building_number", how="left")
    test["date_time"] = pd.to_datetime(test["date_time"], format="%Y%m%d %H")

    # date time feature 생성
    test["hour"] = test["date_time"].dt.hour
    test["day"] = test["date_time"].dt.day
    test["month"] = test["date_time"].dt.month
    test["weekday"] = test["date_time"].dt.weekday

    test_x = test.drop(columns=[*cfg.features.drop_features, "date_time"])
    test_x = categorize_test_features(cfg, test_x)

    return test_x
