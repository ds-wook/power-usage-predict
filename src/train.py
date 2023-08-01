from __future__ import annotations

import warnings
from pathlib import Path

import flash
import hydra
import torch
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_forecasting.data import NaNLabelEncoder

from data.dataset import load_train_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, LinearBoostingTrainer, XGBoostTrainer


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
            lgb_trainer.save_model(save_path / cfg.models.results)

        elif cfg.models.name == "catboost":
            # load dataset
            train = load_train_dataset(cfg)
            # train model
            cb_trainer = CatBoostTrainer(config=cfg)
            cb_trainer.train_cross_validation(train)

            # save model
            cb_trainer.save_model(save_path / cfg.models.results)

        elif cfg.models.name == "xgboost":
            # load dataset
            train = load_train_dataset(cfg)
            # train model
            xgb_trainer = XGBoostTrainer(config=cfg)
            xgb_trainer.train_cross_validation(train)

            # save model
            xgb_trainer.save_model(save_path / cfg.models.results)

        elif cfg.models.name == "linearboosting":
            # load dataset
            train = load_train_dataset(cfg)
            # train model
            lb_trainer = LinearBoostingTrainer(config=cfg)
            lb_trainer.train_cross_validation(train)

            # save model
            lb_trainer.save_model(save_path / cfg.models.results)

        elif cfg.models.name == "n_beats":
            # load dataset
            data = load_train_dataset(cfg)
            training_cutoff = data["time_idx"].max() - cfg.models.max_prediction_length

            datamodule = TabularForecastingData.from_data_frame(
                time_idx="time_idx",
                target=cfg.data.target,
                group_ids=["building_number"],
                categorical_encoders={"building_number": NaNLabelEncoder(add_nan=True).fit(data["building_number"])},
                time_varying_unknown_reals=[cfg.data.target],
                # max_encoder_length=cfg.models.max_encoder_length,
                max_prediction_length=cfg.models.max_prediction_length,
                train_data_frame=data[lambda x: x.time_idx <= training_cutoff],
                val_data_frame=data[lambda x: x.time_idx > training_cutoff],
                batch_size=cfg.models.batch_size,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )

            # 2. Build the task
            model = TabularForecaster(
                datamodule.parameters,
                backbone=cfg.models.params.backbone,
                backbone_kwargs={"widths": [32, 512], "backcast_loss_ratio": 0.1},
                optimizer=cfg.models.params.optimizer,
                learning_rate=cfg.models.params.lr,
            )
            # 3. Create the trainer and train the model
            trainer = flash.Trainer(
                max_epochs=cfg.models.max_epochs,
                gpus=torch.cuda.device_count(),
                gradient_clip_val=cfg.models.gradient_clip_val,
            )
            trainer.fit(model, datamodule=datamodule)
            # 5. Save the model!
            trainer.save_checkpoint(save_path / cfg.models.results)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
