import matplotlib.pyplot as plt
import numpy as np
import polars.selectors as cs


def plot_stochastic(z, sample: np.ndarray, label, ax):
    std = sample.std(axis=-1)
    mean = sample.mean(axis=-1)
    ax.fill_betweenx(z, mean - 2 * std, mean + 2 * std, color="cyan")
    ax.plot(mean, z, label=label, color="blue")


def plot_mc_results(depths, stochastic_predictions, observed_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].set_ylabel("Z")

    for i, label in enumerate(["Temperature", "Density"]):
        plot_stochastic(-depths, stochastic_predictions[:, i, :], "Predictions", ax[i])
        ax[i].plot(observed_data[:, i], -depths, label="Observed", color="orange")
    return ax

def plot_train_frac_table(table, embedding_versions, labels, **kwargs):
    table = table.select("model", "version", "train_frac", "rmse.per_sample.temperature").unnest("rmse.per_sample.temperature")
    means = table.select(~cs.by_name("std")).pivot(on=["model", "version"], values=["mean"])
    stds = table.select(~cs.by_name("mean")).pivot(on=["model", "version"], values=["std"])
    fig, axs = plt.subplots(len(embedding_versions), 1, squeeze=False, constrained_layout=True, sharex=True, **kwargs)
    for version, ax in zip(embedding_versions, axs):
        ax[0].set_title(version)
        for label in labels:
            ax[0].errorbar(means["train_frac"], means[f'{{"{label}","{version}"}}'],
                           yerr=stds[f'{{"{label}","{version}"}}'], label=label)
    axs[-1, 0].legend()