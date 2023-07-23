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
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(np.log1p(train["강수량(mm)"]))
# %%
import flash
import pandas as pd
import torch
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData

import pandas as pd  # noqa: E402
from pytorch_forecasting.data import NaNLabelEncoder  # noqa: E402
from pytorch_forecasting.data.examples import generate_ar_data  # noqa: E402

# %%
# Example based on this tutorial: https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html
# 1. Create the DataModule
data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")

max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

print(training_cutoff)
data.head()
# %%
data["time_idx"].unique()

# %%
data.shape
# %%
datamodule = TabularForecastingData.from_data_frame(
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    # only unknown variable is "value" - and N-Beats can also not take any additional variables
    time_varying_unknown_reals=["value"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    train_data_frame=data[lambda x: x.time_idx <= training_cutoff],
    # validate on the last sequence
    val_data_frame=data[lambda x: x.time_idx > training_cutoff - max_encoder_length],
    batch_size=32,
)
# %%
# 2. Build the task
model = TabularForecaster(
    datamodule.parameters,
    backbone="n_beats",
    backbone_kwargs={"widths": [32, 512], "backcast_loss_ratio": 0.1},
)
# %%
# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count(), gradient_clip_val=0.01)
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions
datamodule = TabularForecastingData.from_data_frame(
    predict_data_frame=data[lambda x: x.time_idx > training_cutoff - max_encoder_length],
    parameters=datamodule.parameters,
    batch_size=32,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# %%
predictions = [prediction[0]["prediction"].cpu().detach().numpy() for prediction in predictions]

print(predictions)
# %%
from itertools import chain

list(chain(*predictions))

# %%
