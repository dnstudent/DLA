from math import floor

import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples
from ..datasets.windowed import WindowedDataset

class GappedKFold(BaseCrossValidator):
    def __init__(self, n_splits, gap_size):
        super().__init__()
        self.n_splits = n_splits
        self.gap_size = gap_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

class MaxSizeKFold(BaseCrossValidator):
    def __init__(self, n_splits, max_val_frac):
        super().__init__()
        self.n_splits = n_splits
        self.max_val_frac = max_val_frac

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None, seed=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        max_val_size = int(n_samples * self.max_val_frac)

        val_size = min(max_val_size, n_samples // self.n_splits)
        for i in range(self.n_splits):
            val_indices = indices[i*val_size:(i+1)*val_size]
            train_indices = np.concatenate([indices[:i*val_size],indices[(i+1)*val_size:]])
            yield train_indices, val_indices
