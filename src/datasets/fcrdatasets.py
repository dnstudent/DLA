import numpy as np
import polars as pl
import polars.selectors as cs

from .tools import windowed, periodic_day, density


def fcr_drivers_table(drivers_csv_path):
    return pl.read_csv(drivers_csv_path, schema_overrides={"time": pl.Date})

def fcr_autoencoder_dataset(drivers_csv_path, window_size: int):
    data = fcr_drivers_table(drivers_csv_path).with_columns(
        periodic_day(pl.col("time")).alias("date_components")
    ).unnest("date_components")
    return windowed(data.drop("time").to_numpy().astype(np.float32), window_size), None, windowed(data["time"].to_numpy(), window_size)

def fcr_full_table(csv_path):
    return pl.read_csv(csv_path).sort("time", "depth")

def fcr_spatiotemporal_dataset(table, depth_steps = 28):
    if type(table) == str:
        table = fcr_full_table(table)
    table = table.filter(pl.col("temp_observed").is_not_null()).sort("time", "depth").drop("time")
    return (
        table.drop(["temp_observed"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, len(table.columns) - 1)),
        table.select(["temp_observed"]).with_columns(density(pl.col("temp_observed")).alias("density")).select(["temp_observed", "density"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, 2))
    )