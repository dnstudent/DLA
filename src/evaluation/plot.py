import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

from .analysis import table_from_results


def plot_stochastic(z, sample: np.ndarray, label, ax):
    std = sample.std(axis=-1)
    mean = sample.mean(axis=-1)
    ax.fill_betweenx(z, mean - 2 * std, mean + 2 * std, color="lightsteelblue")
    ax.plot(mean, z, label=label, color="cornflowerblue")

def plot_samples(z, samples: np.ndarray, label, ax):
    ax.plot(samples, z, label=label)

def plot_mc_results(depths, stochastic_predictions, observed_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].set_ylabel("Z")

    for i, label in enumerate(["Temperature", "Density"]):
        plot_stochastic(-depths, stochastic_predictions[:, i, :], "Predictions", ax[i])
        ax[i].plot(observed_data[:, i], -depths, label="Observed", color="orange")
    return ax

def plot_mc_sample(depths, stochastic_predictions, observed_data, sample_size, ax=None):
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

def make_figure_6(test_results, model_labels, embedding_version, with_glm, ax=None, **kwargs):
    rs = table_from_results(test_results).select("model", "embedding_version", "train_frac",
                                                 "rmse.per_sample.temperature").unnest(
        "rmse.per_sample.temperature").with_columns(pl.col("embedding_version").fill_null("none"))
    means = rs.select(~cs.by_name("std")).pivot(on=["model", "embedding_version"], values=["mean"])
    stds = rs.select(~cs.by_name("mean")).pivot(on=["model", "embedding_version"], values=["std"])
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    version = embedding_version if embedding_version else "none"
    ax.set_title(f"{version} embedding, {'with glm' if with_glm else 'without glm'}")
    for label in model_labels:
        ax.errorbar(means["train_frac"], means[f'{{"{label}","{version}"}}'],
                       yerr=stds[f'{{"{label}","{version}"}}'], label=label, capsize=5.0)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(which="both", visible=True)
    ax.legend()
    return ax

def make_figure_7_a(sample, mc_models, model_labels, depths, y_test, axs):
    if axs is None:
        fig, axs = plt.subplots(len(mc_models.items()), 2, figsize=(10, 15), sharex="col", sharey=True,
                            squeeze=False, constrained_layout=True)
    # fig.tight_layout()
    for sample, ax, model_label in zip(list(sample.values()), axs, model_labels):
        plot_mc_results(depths, sample, y_test, ax)
        ax[0].set_title(model_label)
    axs[-1, 0].set_xlabel("Temperature")
    axs[-1, 1].set_xlabel("Density")
    return axs

def make_figure_7_b(sample, mc_models, model_labels, depths, y_test, axs):
    if axs is None:
        fig, axs = plt.subplots(len(mc_models.items()), 2, figsize=(10, 25), sharex="col", sharey=True,
                            squeeze=False, constrained_layout=True)
    # fig.tight_layout()
    for sample, ax, model_label in zip(list(sample.values()), axs, model_labels):
        take_idxs = np.arange(sample.shape[-1], dtype=np.int32)
        np.random.shuffle(take_idxs)
        take_idxs = take_idxs[:10]
        for i in range(2):
            ax[i].plot(sample[:, i, take_idxs].transpose(), -depths, linewidth=0.5)
            ax[i].scatter(y_test[:, i], -depths)
        ax[0].set_title(model_label)
    axs[-1, 0].set_xlabel("Temperature")
    axs[-1, 1].set_xlabel("Density")
    return axs

def make_table_1(test_results):
    return (table_from_results(test_results)
            .select("model", "embedding_version", "train_frac", "rmse.per_sample.temperature", "rmse.mean.temperature")
            .unnest("rmse.per_sample.temperature", "rmse.mean.temperature")
            .with_columns(pl.col("embedding_version").fill_null("none")))