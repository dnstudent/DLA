from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import polars.selectors as cs
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from scipy.io import loadmat

from .common import make_autoencoder_dataset, make_autoencoder_split_dataset, make_spatiotemporal_dataset, \
    make_spatiotemporal_split_dataset, make_spatiotemporal_dataset_v2, density
from .tools import periodic_day
from .transformers import scale_wds, StandardScaler


def read_drivers_table(drivers_path) -> pl.DataFrame:
    return pl.read_csv(drivers_path, schema_overrides={"time": pl.Date}).rename({"time": "date"}).sort("date")


autoencoder_dataset = make_autoencoder_dataset(read_drivers_table)
autoencoder_split_dataset = make_autoencoder_split_dataset(autoencoder_dataset)


class AutoencoderDataModule(LightningDataModule):
    def __init__(
            self, drivers_csv_path: str, n_timesteps: int, batch_size: int, test_frac: float = 0.05, seed: int = 42,
            shuffle: bool = False
            ):
        super().__init__()
        self.drivers_table = read_drivers_table(drivers_csv_path)
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.test_frac = test_frac
        self.train_ds, self.test_ds, self.predict_ds = autoencoder_split_dataset(
            drivers_csv_path, n_timesteps, test_frac, seed, shuffle, ordinal_day=False
            )
        self.fcr_valid = None
        self.window_scaler = StandardScaler()
        self.window_scaler.fit(self.train_ds.unique_entries(0))
        self.train_ds = scale_wds(self.window_scaler, self.train_ds)
        self.test_ds = scale_wds(self.window_scaler, self.test_ds)
        self.predict_ds = scale_wds(self.window_scaler, self.predict_ds)
        self.timesteps = self.drivers_table["date"][n_timesteps - 1:]

    @property
    def n_raw_features(self):
        return self.drivers_table.shape[1] - 1  # without time and doys

    @property
    def n_dataset_features(self):
        return self.drivers_table.shape[-1] + 1

    def setup(self, stage: Optional[str] = None):
        return

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # assert self.fcr_valid is not None
        if self.fcr_valid is None:
            self.fcr_valid = self.train_ds
        return DataLoader(self.fcr_valid, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        # assert self.test_ds is not None
        if self.test_ds is None or self.test_frac == 0.:
            self.test_ds = self.train_ds
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.predict_ds is not None
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False)


def full_table(ds_dir, embedded_features_csv_path, with_glm, ordinal_day):
    ds_dir = Path(ds_dir)
    time_features = read_drivers_table(ds_dir / 'FCR_2013_2018_Drivers.csv')
    if ordinal_day:
        time_features = time_features.with_columns(day=pl.col("date").dt.ordinal_day())
    else:
        time_features = time_features.with_columns(
            periodic_day(pl.col("date")).alias("date_components")
        ).unnest("date_components")
    # glm_temperatures = pl.read_csv(ds_dir / 'FCR_2013_2018_GLM_output.csv', schema_overrides={"time": pl.Date}).rename({"time": "date"})
    actual_temperatures = pl.read_csv(
        ds_dir / 'FCR_2013_2018_Observed_with_GLM_output.csv', schema_overrides={"time": pl.Date}
    ).rename({"time": "date"})
    if embedded_features_csv_path:
        temporal_embedded = pl.read_csv(embedded_features_csv_path, schema_overrides={"time": pl.Date}).select("time", cs.matches(r"e\d+")).rename({"time": "date"})
    else:
        temporal_embedded = pl.DataFrame([pl.Series("date", [], dtype=pl.Date)])
    # computing the growing degree days feature (wrong? shouldn't it be daily?)
    min_temp = 5
    time_features = time_features.with_columns(
        (pl.max_horizontal(
            min_temp, pl.col("AirTemp")
        ).mean().over(pl.col("date").dt.year()) - min_temp).alias("growing_degree_days")
    ).join(temporal_embedded, on="date", how="left").filter(pl.all_horizontal(cs.by_name(["e1", "AirTemp"], require_all=False)).is_not_null())

    table = (actual_temperatures.join(time_features, on="date", how="inner").sort("date", "depth").rename(
        {"temp_observed": "temp", "temp_glm": "glm_temp"}
        ))
    if with_glm:
        return table
    return table.drop("glm_temp")

def transform_original_y(y):
    temps = y[..., 0]
    densities = density(temps)
    return np.stack([temps, densities], axis=-1)

def original_split_dataset(ds_dir):
    ds_dir = Path(ds_dir)
    data = loadmat(str(ds_dir / "ROA_temporal_mendota_train_test_split_4_year_train_new.mat"), squeeze_me=True, variable_names=['train_X','train_Y_true','test_X','test_Y_true'])
    train_y = transform_original_y(data['train_Y_true'])
    test_y = transform_original_y(data['test_Y_true'])
    return data["train_X"].astype(np.float32), data["test_X"].astype(np.float32), np.random.normal(loc=0.0, scale=1.0, size=(len(train_y), 1, 1)).astype(np.float32), np.random.normal(loc=0.0, scale=1.0, size=(len(test_y), 1, 1)).astype(np.float32), train_y.astype(np.float32), test_y.astype(np.float32), None, None

spatiotemporal_dataset = make_spatiotemporal_dataset(full_table, read_drivers_table, depth_steps=28, time_steps=7)
spatiotemporal_split_dataset = make_spatiotemporal_split_dataset(spatiotemporal_dataset, split=188)

spatiotemporal_dataset_v2 = make_spatiotemporal_dataset_v2(full_table, read_drivers_table, depth_steps=28, time_steps=7)
spatiotemporal_split_dataset_v2 = make_spatiotemporal_split_dataset(spatiotemporal_dataset_v2, split=188)
