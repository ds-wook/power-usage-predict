from __future__ import annotations

import warnings
from pathlib import Path

import flash
import hydra
import torch
import torch.nn as nn
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_forecasting.data import NaNLabelEncoder
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
from tqdm import tqdm

from data.dataset import create_data_loader, load_train_dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from models.net import LSTM, EarlyStoppingCallback


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(get_original_cwd()) / cfg.models.path / cfg.models.name

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

        elif cfg.models.name == "lstm":
            X_train, y_train, X_valid, y_valid = load_train_dataset(cfg)
            X_train = X_train.fillna(0)
            X_valid = X_valid.fillna(0)
            train_loader = create_data_loader(X_train.to_numpy(), cfg.models.window_size, cfg.models.batch_size)
            valid_loader = create_data_loader(X_valid.to_numpy(), cfg.models.window_size, cfg.models.batch_size)
            loss_stat = {"train": [], "validation": []}
            early_stopping_callback = EarlyStoppingCallback(0.001, cfg.models.patience)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = LSTM(
                input_size=cfg.models.input_size,
                hidden_size=cfg.models.hidden_size,
                num_layers=cfg.models.num_layers,
                output_size=cfg.models.output_size,
            )
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.models.lr)
            model.to(device)

            for progress in tqdm(range(1, cfg.models.num_epochs + 1), leave=False):
                train_epoch_loss = 0
                train_epoch_smape = 0
                model.train()

                # We loop over training dataset using batches (we use DataLoader to load data with batches)
                for X_train_batch, y_train_batch in train_loader:
                    X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward pass ->>>>
                    y_train_pred = model(X_train_batch)

                    # Find Loss and backpropagation of gradients
                    train_loss = criterion(y_train_pred, y_train_batch)
                    train_smape = symmetric_mean_absolute_percentage_error(y_train_pred.squeeze(), y_train_batch)
                    # backward <-------
                    train_loss.backward()

                    # Update the parameters (weights and biases)
                    optimizer.step()

                    train_epoch_loss += train_loss.item()
                    train_epoch_smape += train_smape.item()
                #  Then we validate our model - concept is the same
                with torch.no_grad():
                    val_epoch_loss = 0
                    val_epoch_smape = 0
                    model.eval()

                    for X_val_batch, y_val_batch in valid_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                        y_val_pred = model(X_val_batch)

                        val_loss = criterion(y_val_pred, y_val_batch)
                        val_smape = symmetric_mean_absolute_percentage_error(y_val_pred.squeeze(), y_val_batch)

                        val_epoch_loss += val_loss.item()
                        val_epoch_smape += val_smape.item()

                    # end of validation loop
                    early_stopping_callback(val_epoch_loss / len(valid_loader))

                    if early_stopping_callback.stop_training:
                        break

                    loss_stat["train"].append(train_epoch_loss / len(train_loader))
                    loss_stat["validation"].append(val_epoch_loss / len(valid_loader))

                print(
                    f"Epoch {progress} train loss: {train_epoch_loss / len(train_loader)}"
                    + f" train smape: {train_epoch_smape}"
                )
                print(
                    f"Epoch {progress} validation loss: {val_epoch_loss / len(valid_loader)}"
                    + f" validation smape: {val_epoch_smape}"
                )

            torch.save(model.state_dict(), save_path / cfg.models.results)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
