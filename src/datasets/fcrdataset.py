from .tools import windowed

import numpy as np
import polars as pl

def _periodic_day(date: pl.Expr) -> pl.Expr:
    days_normalized = (date - date.dt.truncate("1y")).dt.total_days() / (pl.date(date.dt.year(), 12, 31) - date.dt.truncate("1y")).dt.total_days()
    days_normalized = days_normalized * 2 * np.pi
    return pl.struct(sin_day = days_normalized.sin(), cos_day = days_normalized.cos())

def fcr_table(drivers_csv_path):
    return pl.read_csv(drivers_csv_path, schema_overrides={"time": pl.Date})

def fcr_dataset(drivers_csv_path, window_size: int):
    data = fcr_table(drivers_csv_path).with_columns(
        _periodic_day(pl.col("time")).alias("date_components")
    ).unnest("date_components")
    return windowed(data.drop("time").to_numpy().astype(np.float32), window_size), None, windowed(data["time"].to_numpy(), window_size)
