import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class FeatureEngineering:
    def __init__(self, config: DictConfig, df: pd.DataFrame):
        self.config = config

        df = self._add_time_features(df)
        df = self._add_features(df)
        df = self._fill_missing_features(df)
        df = self._add_solar_features(df)

        self.df = df

    def get_train_preprocessed(self):
        self.df = self._categorize_train_features(self.df)
        return self.df

    def get_test_preprocessed(self):
        self.df = self._categorize_test_features(self.df)
        return self.df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        df["date_time"] = pd.to_datetime(df["date_time"], format="%Y%m%d %H")
        df["hour"] = df["date_time"].dt.hour
        df["day"] = df["date_time"].dt.day
        df["month"] = df["date_time"].dt.month
        df["weekday"] = df["date_time"].dt.weekday
        df["weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        df["total_area"] = np.log1p(df["total_area"])
        df["cooling_area"] = np.log1p(df["cooling_area"])

        weather_features = ["temperature", "rainfall", "windspeed", "humidity"]
        df_num_agg = df.groupby(["building_number", "day", "month"])[weather_features].agg(["mean"])
        df_num_agg.columns = ["_".join(col) for col in df_num_agg.columns]

        df = pd.merge(df, df_num_agg, on=["building_number", "day", "month"], how="left")

        return df

    def _fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing features
        Args:
            df: dataframe
        Returns:
            dataframe
        """

        for col in tqdm(["rainfall", "windspeed", "humidity"], leave=False):
            df[col] = df[col].fillna(df.groupby("building_number")[col].transform("mean"))

        return df

    def _add_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add latitude features
        Args:
            df: dataframe
        Returns:
            dataframe
        """

        df["solarHour"] = (df["hour"] - 12) * 15
        df["solarDec"] = -23.45 * np.cos(np.deg2rad(360 * (df["day"] + 10) / 365))

        return df

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

        for cat_feature in tqdm(self.config.features.categorical_features, leave=False):
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

        for cat_feature in tqdm(self.config.features.categorical_features, leave=False):
            le = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
            test[cat_feature] = le.transform(test[cat_feature])

        return test
