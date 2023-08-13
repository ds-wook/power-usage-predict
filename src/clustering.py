from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from features.engine import FeatureEngineering


def cluster_features(df: pd.DataFrame) -> dict[int, int]:
    scaler = StandardScaler()

    d_list = []
    for i in range(1, 100 + 1):
        d = df[df["building_number"] == i].copy()
        d["power_consumption"] = scaler.fit_transform(d["power_consumption"].values.reshape(-1, 1))
        d_ = d.groupby(["weekday", "hour"])["power_consumption"].mean().unstack()
        d_list.append(d_.to_numpy())

    # 정규화된 시간-요일별 전력사용량을 이미지처럼 저장
    d_list = np.array(d_list)
    d_list_ = d_list.reshape(100, -1)
    model = KMeans(init="k-means++", n_clusters=10, random_state=0)
    model.fit(d_list_)
    cluster = model.labels_
    cluster_map = {i + 1: c for i, c in enumerate(cluster)}

    return cluster_map


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
        cluster_map = cluster_features(train)

        with open(Path(get_original_cwd()) / cfg.data.encoder / "cluster_map.pkl", "wb") as f:
            pickle.dump(cluster_map, f)


if __name__ == "__main__":
    _main()
