from pathlib import Path
from typing import Optional

import polars as pl
import polars.selectors as cs
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .common import make_autoencoder_dataset, make_autoencoder_split_dataset, make_spatiotemporal_dataset, \
    make_spatiotemporal_split_dataset, make_spatiotemporal_dataset_v2
from .tools import periodic_day, density
from .transformers import scale_wds, StandardScaler


def read_drivers_table(drivers_path) -> pl.DataFrame:
    return pl.read_csv(drivers_path, schema_overrides={"time": pl.Date}).rename({"time": "date"}).sort("date")

autoencoder_dataset = make_autoencoder_dataset(read_drivers_table)
autoencoder_split_dataset = make_autoencoder_split_dataset(autoencoder_dataset)

class AutoencoderDataModule(LightningDataModule):
    def __init__(self, drivers_csv_path: str, n_timesteps: int, batch_size: int, test_frac: float = 0.05, seed: int = 42, shuffle: bool = False):
        super().__init__()
        self.drivers_table = read_drivers_table(drivers_csv_path)
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.test_frac = test_frac
        self.train_ds, self.test_ds, self.predict_ds = autoencoder_split_dataset(drivers_csv_path, n_timesteps, test_frac, seed, shuffle)
        self.fcr_valid = None
        self.window_scaler = StandardScaler()
        self.window_scaler.fit(self.train_ds.unique_entries(0))
        self.train_ds = scale_wds(self.window_scaler, self.train_ds)
        self.test_ds = scale_wds(self.window_scaler, self.test_ds)
        self.predict_ds = scale_wds(self.window_scaler, self.predict_ds)
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
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.fcr_valid is not None
        return DataLoader(self.fcr_valid, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        assert self.test_ds is not None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.predict_ds is not None
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False)

def full_table(ds_dir, embedded_features_csv_path):
    ds_dir = Path(ds_dir)
    time_features = read_drivers_table(ds_dir / 'FCR_2013_2018_Drivers.csv').with_columns(
        periodic_day(pl.col("date")).alias("date_components")
    ).unnest("date_components")
    glm_temperatures = pl.read_csv(ds_dir / 'FCR_2013_2018_GLM_output.csv', schema_overrides={"time": pl.Date}).rename({"time": "date"})
    actual_temperatures = pl.read_csv(ds_dir / 'FCR_2013_2018_Observed_with_GLM_output.csv',
                                      schema_overrides={"time": pl.Date}).rename({"time": "date"})
    if embedded_features_csv_path:
        temporal_embedded = pl.read_csv(embedded_features_csv_path, schema_overrides={"time": pl.Date}).select("time", cs.matches(r"e\d+")).rename({"time": "date"})
    else:
        temporal_embedded = pl.DataFrame([pl.Series("date", [], dtype=pl.Date)])
    # computing the growing degree days feature
    min_temp = 5
    time_features = time_features.with_columns(
        (
                pl.max_horizontal(
                    min_temp,
                    pl.col("AirTemp")
                )
                .mean().over(pl.col("date")) - min_temp
        ).alias("growing_degree_days")
    ).join(temporal_embedded, on="date", how="left").filter(pl.all_horizontal(cs.by_name(["e1", "AirTemp"], require_all=False)).is_not_null())
    all_temperatures = (
        glm_temperatures.with_columns(
            (pl.col("depth") * 100).cast(pl.Int32)
        )
        .join(
            actual_temperatures.with_columns(
                (pl.col("depth") * 100).cast(pl.Int32)
            ), on=["date", "depth"], how="left").drop("temp_glm")
        .rename({"temp": "glm_temp"})
        .with_columns(pl.col("depth").truediv(100))
    )
    return (
        all_temperatures
        .join(time_features, on="date", how="inner")
        .sort("date", "depth")
        .rename({"temp_observed": "temp"})
    )

spatiotemporal_dataset = make_spatiotemporal_dataset(full_table, read_drivers_table, depth_steps=28, time_steps=7)
spatiotemporal_split_dataset = make_spatiotemporal_split_dataset(spatiotemporal_dataset, split=189)


spatiotemporal_dataset_v2 = make_spatiotemporal_dataset_v2(full_table, read_drivers_table, depth_steps=28, time_steps=7)
spatiotemporal_split_dataset_v2 = make_spatiotemporal_split_dataset(spatiotemporal_dataset_v2, split=189)