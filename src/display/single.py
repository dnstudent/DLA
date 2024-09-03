from typing import List
from src.models.mcdropout import MCSampler
import matplotlib.pyplot as plt
import numpy as np

def plot_stochastic(x, sample, label, ax):
    quantiles = np.quantile(sample, [0.05, 0.95], axis=-1)
    ax.fill_between(x, quantiles[0], quantiles[1])
    ax.plot(sample.mean(axis=-1), label=label)

def plot_mc_results(stochastic_predictions, observed_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2)
    x = np.arange(len(stochastic_predictions))
    for i, label in enumerate(["Temperature", "Density"]):
        ax[i].set_title(label)
        plot_stochastic(x, stochastic_predictions[:, i, :], "Predictions", ax[i])
        ax[i].plot(observed_data[:, i], label="Observed")
    return ax

def plot_mc_performance(model, day_data, sample_size, ax=None):
    mc_model = MCSampler(model, sample_size)

    return