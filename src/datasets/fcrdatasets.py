import numpy as np
import polars as pl

from .tools import windowed, periodic_day, density


def fcr_drivers_table(drivers_csv_path):
    return pl.read_csv(drivers_csv_path, schema_overrides={"time": pl.Date})

def fcr_autoencoder_dataset(drivers_csv_path, window_size: int):
    data = fcr_drivers_table(drivers_csv_path).with_columns(
        periodic_day(pl.col("time")).alias("date_components")
    ).unnest("date_components")
    return windowed(data.drop("time").to_numpy().astype(np.float32), window_size), None, windowed(data["time"].to_numpy(), window_size)

def fcr_spatiotemporal_dataset(csv_path, depth_steps = 28):
    table = pl.read_csv(csv_path).sort("time", "depth").filter(pl.col("temp_observed").is_not_null()).drop("time", "depth")
    return (
        table.drop(["temp_observed", "glm_temp"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, len(table.columns) - 2)),
        table.select(["temp_observed"]).with_columns(density(pl.col("temp_observed").alias("density"))).to_numpy().astype(np.float32).reshape((-1, depth_steps, 2))
    )