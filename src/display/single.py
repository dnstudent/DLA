from typing import List
from src.models.mcdropout import MCSampler
import matplotlib.pyplot as plt
import numpy as np

def plot_stochastic(z, sample: np.ndarray, label, ax):
    std = sample.std(axis=-1)
    mean = sample.mean(axis=-1)
    ax.fill_betweenx(z, mean - std, mean + std, color="cyan")
    ax.plot(mean, z, label=label, color="blue")


def plot_mc_results(depths, stochastic_predictions, observed_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].set_ylabel("Z")

    for i, label in enumerate(["Temperature", "Density"]):
        plot_stochastic(-depths, stochastic_predictions[:, i, :], "Predictions", ax[i])
        ax[i].plot(observed_data[:, i], -depths, label="Observed", color="orange")
    return ax

def plot_mc_performance(model, day_data, sample_size, ax=None):
    mc_model = MCSampler(model, sample_size)

    return