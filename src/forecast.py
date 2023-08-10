from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping

from data.dataset import load_test_dataset, load_train_dataset


def forecast(cfg: DictConfig, ckpt: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    # load model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    max_encoder_length = best_tft.dataset_parameters["max_encoder_length"]
    # max_prediction_length = best_tft.dataset_parameters["max_prediction_length"]

    # use 5 weeks of training data at the end
    encoder_data = train_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    target_cols = [c for c in test_df.columns if cfg.data.target in c]

    # get last entry from training data
    last_data = train_df.iloc[[-1]]

    for c in target_cols:
        test_df.loc[:, c] = last_data[c].item()

    decoder_data = test_df

    # combine encoder and decoder data. decoder data is to be predicted
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

    preds = new_raw_predictions["prediction"].squeeze()
    sub_df = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.output.submission)

    sub_df["answer"] = preds.cpu().numpy().flatten()

    # save predction to a csv file
    outfn = Path(get_original_cwd()) / cfg.output.path / f"{cfg.models.results}.csv"

    sub_df.to_csv(outfn, index=False)


@hydra.main(config_path="../config/", config_name="forecast")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # load dataset
        train = load_train_dataset(cfg)
        training_cutoff = train["time_idx"].max() - cfg.models.max_prediction_length

        training = TimeSeriesDataSet(
            train[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=cfg.data.target,
            group_ids=["building_number"],
            min_encoder_length=cfg.models.max_encoder_length // 2,
            min_prediction_length=1,
            target_normalizer=GroupNormalizer(groups=["building_number"], transformation="softplus"),
            time_varying_unknown_reals=[cfg.data.target],
            time_varying_known_reals=["time_idx", "hour", "day", "month", "weekday", "weekend", "sin_time", "cos_time"],
            max_encoder_length=cfg.models.max_encoder_length,
            max_prediction_length=cfg.models.max_prediction_length,
            add_relative_time_idx=True,  # add as feature
            add_target_scales=True,  # add as feature
            add_encoder_length=True,  # add as feature
        )

        validation = TimeSeriesDataSet.from_dataset(training, train, predict=True, stop_randomization=True)

        train_dataloader = training.to_dataloader(train=True, batch_size=cfg.models.batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=cfg.models.batch_size, num_workers=0)

        pl.seed_everything(42)
        trainer = pl.Trainer(gpus=1, gradient_clip_val=0.01)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=cfg.models.max_epochs,
            gpus=cfg.models.gpus,
            gradient_clip_val=0.01,
            callbacks=[early_stop_callback],
            limit_train_batches=150,
        )

        net = TemporalFusionTransformer.from_dataset(
            dataset=training,
            loss=SMAPE(),
            logging_metrics=[SMAPE()],
            learning_rate=cfg.models.learning_rate,
            log_interval=cfg.models.log_interval,
            log_val_interval=cfg.models.log_val_interval,
            attention_head_size=cfg.models.attention_head_size,
            hidden_size=cfg.models.hidden_size,
            hidden_continuous_size=cfg.models.hidden_continuous_size,
            dropout=cfg.models.dropout,
            reduce_on_plateau_patience=cfg.models.reduce_on_plateau_patience,
            output_size=1,
        )

        trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = best_model.predict(val_dataloader)
        print(predictions.cpu().numpy().reshape(-1, 1).shape)
        print(actuals.cpu().numpy().reshape(-1, 1).shape)
        print(SMAPE(reduction="none")(predictions, actuals).mean(1))

        test = load_test_dataset(cfg)
        forecast(cfg, best_model_path, train, test)


if __name__ == "__main__":
    _main()
