from pathlib import Path
import random
from math import floor
from datetime import date
from typing import Tuple

from .tools import random_split_and_window, windowed

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, DataLoader

import lightning as L

def fcr_table(drivers_csv_path):
    return pl.read_csv(drivers_csv_path, schema_overrides={"time": pl.Date})

def fcr_dataset(drivers_csv_path, window_size: int):
    data = fcr_table(drivers_csv_path) #.drop("time").to_numpy().astype(np.float32)
    return windowed(data.drop("time").to_numpy().astype(np.float32), window_size), None, windowed(data["time"].to_numpy(), window_size)
