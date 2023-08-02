from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping

from data.dataset import load_train_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from models.net import TabNetTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(get_original_cwd()) / cfg.models.path

        if cfg.models.name == "lightgbm":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            lgb_trainer = LightGBMTrainer(config=cfg)
            lgb_trainer.train_cross_validation(train)

            # save model
            lgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "catboost":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            cb_trainer = CatBoostTrainer(config=cfg)
            cb_trainer.train_cross_validation(train)

            # save model
            cb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "xgboost":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            xgb_trainer = XGBoostTrainer(config=cfg)
            xgb_trainer.train_cross_validation(train)

            # save model
            xgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "tabnet":
            # load dataset
            train = load_train_dataset(cfg)

            # train model
            tabnet_trainer = TabNetTrainer(config=cfg)
            tabnet_trainer.train_cross_validation(train)

            # save model
            tabnet_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "n_beats":
            # load dataset
            data = load_train_dataset(cfg)
            training_cutoff = data["time_idx"].max() - cfg.models.max_prediction_length

            training = TimeSeriesDataSet(
                data[lambda x: x.time_idx <= training_cutoff],
                time_idx="time_idx",
                target=cfg.data.target,
                group_ids=["building_number"],
                categorical_encoders={"building_number": NaNLabelEncoder(add_nan=True).fit(data.building_number)},
                time_varying_unknown_reals=[cfg.data.target],
                max_encoder_length=cfg.models.max_encoder_length,
                max_prediction_length=cfg.models.max_prediction_length,
            )

            validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
            batch_size = 128
            train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

            pl.seed_everything(42)
            trainer = pl.Trainer(gpus=1, gradient_clip_val=0.01)
            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
            )
            trainer = pl.Trainer(
                max_epochs=3,
                gpus=1,
                gradient_clip_val=0.01,
                callbacks=[early_stop_callback],
                limit_train_batches=150,
            )

            net = NBeats.from_dataset(
                training,
                learning_rate=1e-3,
                log_interval=10,
                log_val_interval=1,
                weight_decay=1e-2,
                widths=[32, 512],
                backcast_loss_ratio=1.0,
            )

            trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

            best_model_path = trainer.checkpoint_callback.best_model_path
            best_model = NBeats.load_from_checkpoint(best_model_path)
            actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
            predictions = best_model.predict(val_dataloader)

            print(SMAPE()(predictions, actuals))

        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
