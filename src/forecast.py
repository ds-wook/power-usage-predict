from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping

from data.dataset import load_test_dataset, load_train_dataset


def forecast(cfg: DictConfig, ckpt: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    # load model
    best_tft = NBeats.load_from_checkpoint(ckpt)
    max_encoder_length = best_tft.dataset_parameters["max_encoder_length"]
    max_prediction_length = best_tft.dataset_parameters["max_prediction_length"]

    assert max_encoder_length == 5 * 24 * 7 and max_prediction_length == 1 * 24 * 7

    # use 5 weeks of training data at the end
    encoder_data = train_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    test_df[cfg.data.target] = 0

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
            categorical_encoders={"building_number": NaNLabelEncoder(add_nan=True).fit(train.building_number)},
            time_varying_unknown_reals=[cfg.data.target],
            max_encoder_length=cfg.models.max_encoder_length,
            max_prediction_length=cfg.models.max_prediction_length,
        )

        validation = TimeSeriesDataSet.from_dataset(training, train, min_prediction_idx=training_cutoff + 1)

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

        net = NBeats.from_dataset(
            dataset=training,
            loss=SMAPE(),
            logging_metrics=[SMAPE()],
            learning_rate=cfg.models.learning_rate,
            log_interval=cfg.models.log_interval,
            log_val_interval=cfg.models.log_val_interval,
            weight_decay=cfg.models.weight_decay,
            widths=[*cfg.models.widths],
            backcast_loss_ratio=cfg.models.backcast_loss_ratio,
        )

        trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = NBeats.load_from_checkpoint(best_model_path)

        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = best_model.predict(val_dataloader)
        print(predictions.cpu().numpy().reshape(-1, 1).shape)
        print(actuals.cpu().numpy().reshape(-1, 1).shape)
        print(SMAPE()(predictions, actuals))

        test = load_test_dataset(cfg)

        forecast(cfg, best_model_path, train, test)


if __name__ == "__main__":
    _main()
