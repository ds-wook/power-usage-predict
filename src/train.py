from __future__ import annotations

import warnings
from pathlib import Path

import flash
import hydra
import torch
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from models.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_path = Path(get_original_cwd()) / cfg.models.path / cfg.models.name

        if cfg.models.name == "lightgbm":
            # load dataset
            X_train, y_train, X_valid, y_valid = load_train_dataset(cfg)
            # train model
            lgb_trainer = LightGBMTrainer(config=cfg)
            model = lgb_trainer.train(X_train, y_train, X_valid, y_valid)

            # save model
            model.save_model(save_path / cfg.models.results)

        elif cfg.models.name == "n_beats":
            # load dataset
            X_train, X_valid = load_train_dataset(cfg)

            datamodule = TabularForecastingData.from_data_frame(
                time_idx="time_idx",
                target="value",
                group_ids=["series"],
                time_varying_unknown_reals=["value"],
                max_encoder_length=cfg.models.max_encoder_length,
                max_prediction_length=cfg.models.max_prediction_length,
                train_data_frame=X_train,
                val_data_frame=X_valid,
                batch_size=cfg.models.batch_size,
            )

            # 2. Build the task
            model = TabularForecaster(
                datamodule.parameters,
                backbone="n_beats",
                backbone_kwargs={"widths": [32, 512], "backcast_loss_ratio": 0.1},
            )
            # 3. Create the trainer and train the model
            trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count(), gradient_clip_val=0.01)
            trainer.fit(model, datamodule=datamodule)
            # 5. Save the model!
            trainer.save_checkpoint(save_path / cfg.models.results)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
