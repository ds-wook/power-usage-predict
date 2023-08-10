# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit
# %%
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
plt.style.use("fivethirtyeight")


def plot_cv_indices(cv, X, n_splits, lw=10):
    fig, ax = plt.subplots()
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 0.1, -0.1],
        xlim=[0, len(X)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)

    ax.legend([Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))], ["Testing set", "Training set"], loc=(1.02, 0.8))


# %%

train_df = pd.read_csv("../input/power-usage-predict/train.csv")
train_df = train_df.rename(
    columns={
        "건물번호": "building_number",
        "일시": "date_time",
        "기온(C)": "temperature",
        "강수량(mm)": "rainfall",
        "풍속(m/s)": "windspeed",
        "습도(%)": "humidity",
        "일조(hr)": "sunshine",
        "일사(MJ/m2)": "solar_radiation",
        "전력소비량(kWh)": "power_consumption",
    }
)
train_df.drop("num_date_time", axis=1, inplace=True)

# %%

train_df.isna().sum()

# %%
print(train_df["time_idx"].unique())
# %%
# date time feature 생성
train_df["hour"] = train_df["date_time"].dt.hour
train_df["day"] = train_df["date_time"].dt.day
train_df["month"] = train_df["date_time"].dt.month
train_df["year"] = train_df["date_time"].dt.year

np.unique(train_df["hour"] % 12)
# %%
train_df[train_df["building_number"] == 1]
# %%
tqdm.pandas()
train_df["day_hour_mean"] = train_df.progress_apply(
    lambda x: power_mean.loc[
        (power_mean.building_number == x["building_number"])
        & (power_mean.hour == x["hour"])
        & (power_mean.day == x["day"]),
        "power_consumption",
    ].values[0],
    axis=1,
)

# %%
weather_features = ["temperature", "rainfall", "windspeed", "humidity"]
df_trend_agg = train_df.groupby(["building_number", "day", "month"])[weather_features].transform(lambda x: x.diff())

df_trend_agg


# %%


plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train_df["power_consumption"]), bins=30, kde=True)
plt.title("Distribution of Power Consumption")
plt.xlabel("Power Consumption")
plt.ylabel("Frequency")
plt.show()


# %%
def map_date_index(date, min_date):
    return (date - min_date).days


train["일시"] = pd.to_datetime(train["일시"])
min_date = train["일시"].min()

train["time_idx"] = train["일시"].map(lambda date: map_date_index(date, min_date))
# %%
train["time_idx"].unique()
# %%
n_split = 10

tscv = TimeSeriesSplit(n_splits=n_split, gap=1)

plot_cv_indices(tscv, train, n_splits=n_split)
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
train_df["date_time"]
# %%
sns.lineplot(x="date_time", y="power_consumption", data=train_df)
# %%
train = pd.read_csv("../input/power-usage-predict/train_sample.csv")
train.head()
# %%
train.isna().sum()
# %%
df = train.loc[train["heat_index"].isna()]
# %%
df["temperature_f"] = 9 / 5 * df["temperature"] + 32
df["temperature_f"]
# %%
df["heat_index"] = (
    -42.379
    + 2.04901523 * df["temperature_f"]
    + 10.14333127 * df["humidity"]
    - 0.22475541 * df["temperature_f"] * df["humidity"]
    - 0.00683783 * df["temperature_f"] * df["temperature_f"]
    - 0.05481717 * df["humidity"] * df["humidity"]
    + 0.00122874 * df["temperature_f"] * df["temperature_f"] * df["humidity"]
    + 0.00085282 * df["temperature_f"] * df["humidity"] * df["humidity"]
    - 0.00000199 * df["temperature_f"] * df["temperature_f"] * df["humidity"] * df["humidity"]
)
# %%
df["heat_index"]
# %%
df["heat_index"] = (df["heat_index"] - 32) * 5 / 9
df["heat_index"]
# %%
df.loc[df["heat_index"] < 32, "heat_index"] = 0
df.loc[(df["heat_index"] >= 32) & (df["heat_index"] < 41), "heat_index"] = 1
df.loc[(df["heat_index"] >= 41) & (df["heat_index"] < 54), "heat_index"] = 2
df.loc[(df["heat_index"] >= 54) & (df["heat_index"] < 66), "heat_index"] = 3
df.loc[df["heat_index"] >= 66, "heat_index"] = 4

df["heat_index"]
# %%
