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