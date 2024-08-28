import random

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


def contiguous_split(x: np.array, split_start, split_size):
    x_split = x[split_start:split_start+split_size]
    x_l = x[0:split_start]
    x_r = x[split_start+split_size:]
    return x_l, x_split, x_r

def split_and_window(x: np.array, split_start, split_size, window_size):
    if 0 < split_start < 1:
        split_start = int(x.shape[0] * split_start)
    if 0 < split_size < 1:
        split_size = int(x.shape[0] * split_size)
    assert split_start >= 0 and split_size >= 0
    assert split_start + split_size <= x.shape[0]
    x_l, x_split, x_r = contiguous_split(x, split_start, split_size)
    return np.concatenate([windowed(x_l, window_size), windowed(x_r, window_size)]), windowed(x_split, window_size)

def random_split_and_window(x: np.array, split_size, window_size, seed=None):
    if seed is not None:
        random.seed(seed)
    if 0 < split_size < 1:
        split_size = int(x.shape[0] * split_size)
    split_start = random.randint(0, x.shape[0] - split_size - 1)
    return split_and_window(x, split_start, split_size, window_size)

def periodic_day(date: pl.Expr) -> pl.Expr:
    days_normalized = (date - date.dt.truncate("1y")).dt.total_days() / (pl.date(date.dt.year(), 12, 31) - date.dt.truncate("1y")).dt.total_days()
    days_normalized = days_normalized * 2 * np.pi
    return pl.struct(sin_day = days_normalized.sin(), cos_day = days_normalized.cos())

def actual_density(temp):
    return 1 - (temp + 288.9414) * (temp - 3.9863) ** 2 / (508929.2 * (temp + 68.12963))

def density(temp):
    return 1 + actual_density(temp)

def normalize_inputs(x_train, x_test, y_train, y_test):
    x_means, x_stds = x_train.mean(axis=(0,1))[None, None, :], x_train.std(axis=(0,1))[None, None, :]
    y_means, y_stds = y_train.mean(axis=(0,1))[None, None, :], y_train.std(axis=(0,1))[None, None, :]
    return (x_train - x_means) / x_stds, (x_test - x_means) / x_stds, (y_train - y_means) / y_stds, (y_test - y_means) / y_stds, x_means, x_stds, y_means, y_stds

def train_test_split_nd(*arrays, test_size, shuffle=True, random_state=None):
    arrays = [*arrays]
    n_elements = arrays[0].shape[0]
    indices = np.arange(n_elements, dtype=np.int32)
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    if 0 < test_size <= 1:
        test_size = int(n_elements * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    return tuple([(arr[train_indices], arr[test_indices]) for arr in arrays])
