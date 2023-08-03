import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from features.base import BaseDataPreprocessor


class FeatureEngineering(BaseDataPreprocessor):
    def __init__(self, config: DictConfig, df: pd.DataFrame):
        super().__init__(config)

        df = self._add_time_features(df)
        df = self._add_features(df)
        df = self._fill_missing_features(df)
        df = self._add_solar_features(df)
        df = self._add_trend_features(df)
        self.df = df

    def get_train_pipeline(self):
        self.df = self._categorize_train_features(self.df)
        return self.df

    def get_test_pipeline(self):
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
        df.loc[df["date_time"].isin(["2022-06-06", "2022-08-15"]), "weekend"] = 1
        df["sin_time"] = np.sin(2 * np.pi * df.hour / 24)
        df["cos_time"] = np.cos(2 * np.pi * df.hour / 24)

        return df

    # function for feature engineering
    def _cooling_degree_hour(self, xs: np.ndarray) -> list[float]:
        """
        Calculate cooling degree hour
        Args:
            xs: temperature
        Returns:
            cooling degree hour
        """

        ys = [np.sum(xs[: (i + 1)] - 26) if i < 11 else np.sum(xs[(i - 11) : (i + 1)] - 26) for i in range(len(xs))]

        return ys

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
        df["THI"] = 9 / 5 * df["temperature"] - 0.55 * (1 - df["humidity"] / 100) * (9 / 5 * df["humidity"] - 26) + 32

        cdhs = []
        for num in df["building_number"].unique().tolist():
            temp = df[df["building_number"] == num]
            cdh = self._cooling_degree_hour(temp["temperature"].values)
            cdhs += cdh

        else:
            df["CDH"] = cdhs

        weather_features = ["temperature", "windspeed", "humidity"]
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

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        weather_features = ["temperature", "windspeed", "humidity"]

        for col in tqdm(weather_features, leave=False):
            df[f"{col}_trend"] = df.groupby(["building_number", "day", "month"])[col].transform(lambda x: x.diff())
            df[f"{col}_trend"] = df[f"{col}_trend"].fillna(0)

        return df
