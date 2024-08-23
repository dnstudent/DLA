import numpy as np
import polars as pl
import torch
from attr.filters import exclude
from torch.utils.data import Dataset, Subset


class WindowedDataset(Dataset):
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.window_size = times.shape[1]

    def __len__(self):
        return self.times.shape[0]

    def __getitem__(self, idx):
        if type(idx) == np.ndarray:
            idx = idx.copy()
        return self.dataset[idx]

    def _wints_table(self):
        return (
            pl.DataFrame(
                [pl.Series("timestep", self.times.tolist(), dtype=pl.List(pl.Date))]
            )
            .with_row_index("window_idx")
            .explode("timestep")
        )

    def unique_entries(self, field):
        unique_wints = self._wints_table().with_columns(pl.col("timestep").cum_count().over("window_idx").alias("ts_idx") - 1).unique("timestep", keep="any").select("window_idx", "ts_idx")
        window_idxs = unique_wints["window_idx"].to_numpy().copy()
        ts_idxs = unique_wints["ts_idx"].to_numpy().copy()
        return self[:][field][window_idxs, ts_idxs]

    def exclude(self, idxs):
        assert idxs.ndim == 1, "idx must be a 1D numpy array"
        idxs = idxs.copy()
        requested_times = self.times[idxs]
        remaining = (
            self._wints_table()
            .join(
                pl.DataFrame(
                    [pl.Series("timestep", requested_times.flatten(), dtype=pl.Date)]
                ),
                on="timestep",
                how="anti",
            )
            .group_by("window_idx", maintain_order=True)
            .len(name="n")
            .filter(pl.col("n") == self.window_size)
        )
        remaining_windows = remaining["window_idx"].to_numpy().copy()
        remaining_times = self.times[remaining_windows].copy()
        remaining_ds = Subset(self.dataset, remaining_windows)
        return WindowedDataset(remaining_ds, remaining_times)

    def subset(self, idxs):
        assert idxs.ndim == 1, "idx must be a 1D numpy array"
        idxs = idxs.copy()
        requested_times = self.times[idxs]
        requested_ds = Subset(self.dataset, idxs)
        return WindowedDataset(requested_ds, requested_times.copy())

    def train_test_split(self, test_idxs: np.ndarray):
        return self.exclude(test_idxs), self.subset(test_idxs)
