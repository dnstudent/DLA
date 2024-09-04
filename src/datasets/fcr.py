from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from .common import make_autoencoder_dataset, make_autoencoder_split_dataset, make_autoencoder_split_dataset_old, \
    make_autoencoder_dataset_old, make_spatiotemporal_dataset, make_spatiotemporal_split_dataset
from .tools import windowed, periodic_day, density
from .transformers import scale_wds, StandardScaler
from .windowed import WindowedDataset


def read_drivers_table(drivers_path) -> pl.DataFrame:
    return pl.read_csv(drivers_path, schema_overrides={"time": pl.Date}).rename({"time": "date"}).sort("date")

def autoencoder_dataset_old(drivers_table: Union[str, pl.DataFrame], window_size: int):
    if type(drivers_table) is str:
        drivers_table = read_drivers_table(drivers_table)
    return make_autoencoder_dataset_old(drivers_table, window_size)

def autoencoder_split_dataset_old(drivers_path: str, window_size: int, test_frac: float = 0.05, seed: Optional[int] = None, shuffle: bool = False) -> Tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    x, _, t = autoencoder_dataset(drivers_path, window_size)
    return make_autoencoder_split_dataset_old(x, t, test_frac, seed, shuffle)


autoencoder_dataset = make_autoencoder_dataset(read_drivers_table)
autoencoder_split_dataset = make_autoencoder_split_dataset(autoencoder_dataset)

class AutoencoderDataModule(LightningDataModule):
    def __init__(self, drivers_csv_path: str, n_timesteps: int, batch_size: int, test_frac: float = 0.05, seed: int = 42):
        super().__init__()
        self.drivers_table = read_drivers_table(drivers_csv_path)#.with_columns(periodic_day(pl.col("time")).alias("date_components")).unnest("date_components").sort("time")
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.test_frac = test_frac
        self.fcr_train, self.fcr_test, self.fcr_predict = autoencoder_split_dataset(drivers_csv_path, n_timesteps, test_frac, seed)
        self.fcr_valid = None
        self.window_scaler = StandardScaler()
        self.window_scaler.fit(self.fcr_train.unique_entries(0))
        self.fcr_train = scale_wds(self.window_scaler, self.fcr_train)
        self.fcr_test = scale_wds(self.window_scaler, self.fcr_test)
        self.fcr_predict = scale_wds(self.window_scaler, self.fcr_predict)
        self.timesteps = self.drivers_table["time"][n_timesteps-1:]


    @property
    def n_raw_features(self):
        return self.drivers_table.shape[1] - 1 # without time and doys

    @property
    def n_dataset_features(self):
        return self.drivers_table.shape[-1] + 1

    def setup(self, stage: Optional[str] = None):
        return

    def train_dataloader(self):
        return DataLoader(self.fcr_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.fcr_valid is not None
        return DataLoader(self.fcr_valid, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        assert self.fcr_test is not None
        return DataLoader(self.fcr_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.fcr_predict is not None
        return DataLoader(self.fcr_predict, batch_size=self.batch_size, shuffle=False)

def full_table(ds_dir, embedded_features_csv_path):
    ds_dir = Path(ds_dir)
    time_features = pl.read_csv(ds_dir / 'FCR_2013_2018_Drivers.csv', schema_overrides={"time": pl.Date}).with_columns(
        periodic_day(pl.col("time")).alias("date_components")
    ).unnest("date_components")
    glm_temperatures = pl.read_csv(ds_dir / 'FCR_2013_2018_GLM_output.csv', schema_overrides={"time": pl.Date})
    actual_temperatures = pl.read_csv(ds_dir / 'FCR_2013_2018_Observed_with_GLM_output.csv',
                                      schema_overrides={"time": pl.Date})
    if embedded_features_csv_path:
        temporal_embedded = pl.read_csv(embedded_features_csv_path, schema_overrides={"time": pl.Date}).select("time", cs.matches(r"e\d+"))
    else:
        temporal_embedded = pl.DataFrame([pl.Series("time", [], dtype=pl.Date)])
    # computing the growing degree days feature
    min_temp = 5
    time_features = time_features.with_columns(
        (
                pl.max_horizontal(
                    min_temp,
                    pl.col("AirTemp")
                )
                .mean().over(pl.col("time")) - min_temp
        ).alias("growing_degree_days")
    ).join(temporal_embedded, on="time", how="left").filter(pl.all_horizontal(cs.by_name(["e1", "AirTemp"], require_all=False)).is_not_null())
    all_temperatures = (
        glm_temperatures.with_columns(
            (pl.col("depth") * 100).cast(pl.Int32)
        )
        .join(
            actual_temperatures.with_columns(
                (pl.col("depth") * 100).cast(pl.Int32)
            ), on=["time", "depth"], how="left").drop("temp_glm")
        .rename({"temp": "glm_temp"})
        .with_columns(pl.col("depth").truediv(100))
    )
    return (
        all_temperatures
        .join(time_features, on="time", how="inner")
        .filter(pl.col("time") >= pl.date(2013, 5, 21))
        .sort("time", "depth")
        .rename({"time": "date", "temp_observed": "temp"})
    )

def spatiotemporal_dataset_old(ds_dir, embedded_features_csv_path, depth_steps=28):
    table = full_table(ds_dir, embedded_features_csv_path)
    table = table.filter(pl.col("temp_observed").is_not_null()).sort("time", "depth").drop("time")
    return (
        table.drop(["temp_observed"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, len(table.columns) - 1)),
        table.select(["temp_observed"]).with_columns(density(pl.col("temp_observed")).alias("density")).select(["temp_observed", "density"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, 2))
    )

def spatiotemporal_split_dataset_old(ds_dir, embedded_features_csv_path, test_size, depth_steps=28, seed=42, shuffle=False):
    X, y = spatiotemporal_dataset(ds_dir, embedded_features_csv_path, depth_steps)
    return train_test_split(X, y, test_size=test_size, random_state=seed if shuffle else None, shuffle=shuffle)

spatiotemporal_dataset = make_spatiotemporal_dataset(full_table, depth_steps=28)
spatiotemporal_split_dataset = make_spatiotemporal_split_dataset(spatiotemporal_dataset, depth_steps=28)

