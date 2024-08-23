import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from math import floor
import random

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
