# %%
import numpy as np
import pandas as pd


train = pd.read_csv("../input/power-usage-predict/train.csv")
train.head()
# %%
test = pd.read_csv("../input/power-usage-predict/test.csv")
test.head()
# %%
test["건물번호"].unique()
# %%
train["건물번호"].unique()
# %%
building_info = pd.read_csv("../input/power-usage-predict/building_info.csv")
building_info.head()
# %%
train = pd.merge(train, building_info, on="건물번호")
train.head()
# %%
train["태양광용량(kW)"].value_counts()
# %%
