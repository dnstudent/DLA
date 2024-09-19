from math import ceil

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view


def windowed(x, window_size):
    if len(x.shape) > 1:
        window_shape = (window_size, x.shape[1])
        if x.shape[0] >= window_size:
            return sliding_window_view(x, window_shape).squeeze(axis=1).copy()
    else:
        window_shape = (window_size,)
        if x.shape[0] >= window_size:
            return sliding_window_view(x, window_shape).copy()
    return np.array([], dtype=x.dtype).reshape((0,) + window_shape)

# def contiguous_split(x: np.array, split_start, split_size):
#     x_split = x[split_start:split_start+split_size]
#     x_l = x[0:split_start]
#     x_r = x[split_start+split_size:]
#     return x_l, x_split, x_r
#
# def split_and_window(x: np.array, split_start, split_size, window_size):
#     if 0 < split_start < 1:
#         split_start = int(x.shape[0] * split_start)
#     if 0 < split_size < 1:
#         split_size = int(x.shape[0] * split_size)
#     assert split_start >= 0 and split_size >= 0
#     assert split_start + split_size <= x.shape[0]
#     x_l, x_split, x_r = contiguous_split(x, split_start, split_size)
#     return np.concatenate([windowed(x_l, window_size), windowed(x_r, window_size)]), windowed(x_split, window_size)
#
# def random_split_and_window(x: np.array, split_size, window_size, seed=None):
#     if seed is not None:
#         random.seed(seed)
#     if 0 < split_size < 1:
#         split_size = int(x.shape[0] * split_size)
#     split_start = random.randint(0, x.shape[0] - split_size - 1)
#     return split_and_window(x, split_start, split_size, window_size)

def periodic_day(date: pl.Expr) -> pl.Expr:
    days_normalized = (date - date.dt.truncate("1y")).dt.total_days() / (pl.date(date.dt.year(), 12, 31) - date.dt.truncate("1y")).dt.total_days()
    days_normalized = days_normalized * 2 * np.pi
    return pl.struct(sin_day = days_normalized.sin(), cos_day = days_normalized.cos())

def density(temp):
    return 1 - (temp + 288.9414) * (temp - 3.9863) ** 2 / (508929.2 * (temp + 68.12963))

def training_density(temp):
    """???"""
    return 1 + density(temp)

def normalize_inputs(train_arrs, *others):
    var_means = [arr.mean(axis=(0,1))[None, None, :] for arr in train_arrs]
    var_stds = [arr.std(axis=(0,1))[None, None, :] for arr in train_arrs]
    others = [*others]
    train_arrs = [[(train_arr - var_mean) / var_std for train_arr, var_mean, var_std in zip(train_arrs, var_means, var_stds)]]
    other_arrs = [[(test_arr - var_mean) / var_std for test_arr, var_mean, var_std in zip(test_arrs, var_means, var_stds)] for test_arrs in others]
    stats = [var_means, var_stds]
    return train_arrs + other_arrs + stats

def windowed_unique_date_indices(dates: pl.Series, window_size: int):
    return (
        pl.DataFrame(dates)
        .with_columns(window_dates=pl.date_ranges(start=pl.col("date").dt.offset_by(f"-{window_size - 1}d"), end="date"))
        .with_row_index("sample_idx")
        .explode("window_dates")
        .sort("sample_idx", "window_dates")
        .with_columns(window_idx=pl.col("window_dates").cum_count().over("sample_idx"))
        .unique(["date", "window_dates"], maintain_order=True)
        .select("sample_idx", "window_idx")
        .to_numpy()
    )

# def normalize_time_window_inputs(w_train, w_test, t_train, t_test):
#     pass
#
# def train_test_split_nd(*arrays, test_size, shuffle=True, random_state=None):
#     arrays = [*arrays]
#     n_elements = arrays[0].shape[0]
#     indices = np.arange(n_elements, dtype=np.int32)
#     if shuffle:
#         if random_state is not None:
#             np.random.seed(random_state)
#         np.random.shuffle(indices)
#     if 0 < test_size <= 1:
#         test_size = int(n_elements * test_size)
#     train_indices = indices[:-test_size]
#     test_indices = indices[-test_size:]
#     return tuple([(arr[train_indices], arr[test_indices]) for arr in arrays])
#
# def exclude_temporal_indices(dates, window_size, idxs):
#     dates = pl.DataFrame(dates).with_row_index(name="idx")
#     dates = dates.with_columns(
#         window_dates=pl.date_ranges(start=pl.col("date").dt.offset_by(f"-{window_size - 1}x"), end="date"))
#     selected = dates[idxs].sort("idx")
#     left = (
#         dates
#         .join(selected, on="date", how="anti")
#         .explode("window_dates")
#         .join(selected.explode("window_dates"), on="window_dates", how="anti")
#         .group_by("date", "idx")
#         .agg(n=pl.len())
#         .filter(pl.col("n") == window_size)
#         .select("date", "idx").sort("idx")
#     )
#     left_indices = left["idx"].to_numpy()
#     return left_indices
#
# def split_temporal_indices(dates: pl.Series, window_size: np.ndarray, right_frac: float):
#     indices = np.arange(len(dates))
#     split = int(floor((1 - right_frac) * len(dates)))
#     right_indices = indices[split:]
#     return exclude_temporal_indices(dates, window_size, right_indices), right_indices
#
# def split_temporal_rolling(dates: pl.Series, windowed_data: np.ndarray, *arrs, right_frac: float):
#     n_timesteps = len(windowed_data)
#     assert len(dates) == n_timesteps, "There are different timesteps in w and ts"
#     window_size = windowed_data.shape[1]
#     left_indices, right_indices = split_temporal_indices(dates, window_size, right_frac)
#     return [(dates[left_indices], dates[right_indices])] + [(windowed_data[left_indices], windowed_data[right_indices])] + [(arr[left_indices], arr[right_indices]) for arr in arrs]

def their_edge_padding(*arrs, pad_steps, axis):
    arrs = [*arrs]
    if axis == 1:
        pad_width = ((0,0), (pad_steps, 0), (0, 0))
    elif axis == 0:
        pad_width = ((pad_steps, 0), (0, 0), (0, 0))
    else:
        raise NotImplementedError("axis must be 0 or 1")
    return tuple([np.pad(arr, pad_width) for arr in arrs])

def take_frac(*arrs, axis, frac, shuffle, random_state):
    arrs = [*arrs]
    indices = np.arange(arrs[0].shape[axis])
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    take_idx = indices[:int(ceil(len(indices)*frac))]
    return tuple([arr.take(take_idx, axis) for arr in arrs])

def swap_batchtime(*arrs):
    return [np.swapaxes(arr, 0, 1) for arr in [*arrs]]