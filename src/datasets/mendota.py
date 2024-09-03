from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import polars as pl
import polars.selectors as cs
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .common import make_autoencoder_dataset, make_autoencoder_split_dataset
from .tools import density
from .transformers import scale_wds, StandardScaler
from .windowed import WindowedDataset


def read_drivers_table(drivers_feather_path) -> pl.DataFrame:
    return pl.read_ipc(drivers_feather_path, use_pyarrow=True).rename({"time": "date"})

def autoencoder_dataset(drivers_table: Union[str, pl.DataFrame], window_size: int):
    if type(drivers_table) is str:
        drivers_table = read_drivers_table(drivers_table)
    return make_autoencoder_dataset(drivers_table, window_size)

def autoencoder_split_dataset(drivers_feaher_path: str, window_size: int, test_frac: float = 0.05, seed: int = 42) -> Tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    x, _, t = autoencoder_dataset(drivers_feaher_path, window_size)
    return make_autoencoder_split_dataset(x, t, window_size, test_frac, seed)

class AutoencoderDataModule(LightningDataModule):
    def __init__(self, drivers_csv_path: str, n_timesteps: int, batch_size: int, test_frac: float = 0.05, seed: int = 42):
        super().__init__()
        self.drivers_table = read_drivers_table(drivers_csv_path)
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
    time_features = read_drivers_table(ds_dir / "mendota_meteo.feather")
    # GLM data seems incorrect. Many values are identical
    # glm_temperatures = (
    #     pl.read_ipc(ds_dir / "mendota_GLM_uncal_temperatures_anuj.feather")
    #     .with_columns(pl.col("DateTime").dt.date().alias("date"))
    #     .drop("DateTime")
    #     .unpivot(index=["ice", "date"], value_name="glm_temp")
    #     .with_columns(pl.col("variable").str.split_exact("_", 1).struct.rename_fields(["_", "depth"]).struct.field("depth").cast(pl.Float32))
    #     .drop("variable")
    #     .sort("date", "depth")
    # )
    actual_temperatures = pl.read_ipc(ds_dir / 'Mendota_buoy_data_anuj.feather', use_pyarrow=True).rename({"DateTime": "date"}).filter(pl.col("temp").is_not_null())
    valid_dates = actual_temperatures.filter(pl.col("Depth") <= 20.0).group_by("date").len().filter(pl.col("len") >= 23).drop("len")
    valid_depths = actual_temperatures.join(valid_dates, on="date", how="semi").group_by("Depth").len().sort("len").filter(
        pl.col("len") == 1345).drop("len") #1345 days of depth data
    actual_temperatures = actual_temperatures.join(valid_dates, on="date", how="semi").join(valid_depths, on="Depth", how="semi")
    if embedded_features_csv_path:
        temporal_embedded = pl.read_csv(embedded_features_csv_path, schema_overrides={"date": pl.Date}).select("date", cs.matches(r"e\d+"))
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

    return (
        actual_temperatures
        .join(time_features, on="date", how="inner")
        .rename({"Depth": "depth", "temp": "temp_observed"})
        .sort("date", "depth")
    )

def spatiotemporal_dataset(ds_dir, embedded_features_csv_path, depth_steps=21):
    table = full_table(ds_dir, embedded_features_csv_path)
    table = table.filter(pl.col("temp_observed").is_not_null()).sort("date", "depth")#.drop("date")
    return (
        table.drop(["temp_observed"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, len(table.columns) - 1)),
        table.with_columns(density(pl.col("temp_observed")).alias("density")).select(["date", "temp_observed", "density"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, 3))
    )

def spatiotemporal_split_dataset(ds_dir, embedded_features_csv_path, test_size, depth_steps=21, seed=42, shuffle=False):
    X, y = spatiotemporal_dataset(ds_dir, embedded_features_csv_path, depth_steps)
    return train_test_split(X, y, test_size=test_size, random_state=seed if shuffle else None, shuffle=shuffle)