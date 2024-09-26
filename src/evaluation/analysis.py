import numpy as np
import polars as pl
import torch

from src.models.tools import physical_inconsistency as phyincon

def denorm(y, y_means, y_stds):
    if len(y.shape) == 3:
        return y * y_stds + y_means
    return y * y_stds[..., None] + y_means[..., None]

def per_sample_test_rmse(y_true, y_pred_stochastic):
    # Mean on depths (axis=1)
    test_rmse = np.sqrt(np.square(y_true[..., None] - y_pred_stochastic).mean(axis=1))
    # Mean on test_size (axis=0) and statistic on sample_size (axis=-1)
    ttm, dtm = test_rmse.mean(axis=(0, -1))
    tts, dts = test_rmse.std(axis=(0, -1))
    return {"temperature": {"mean": ttm, "std": tts}, "density": {"mean": dtm, "std": dts}}

def their_test_rmse(y_true, y_pred_stochastic):
    means = y_pred_stochastic.mean(axis=3)
    stds = y_pred_stochastic.std(axis=3)

    test_rmse = np.sqrt(np.square(y_true - means).mean(axis=1))
    ttm, dtm = test_rmse.mean(axis=0)
    tts, dts = stds.mean(axis=(0,1))
    return {"temperature": {"mean": ttm, "std": tts}, "density": {"mean": dtm, "std": dts}}


def mean_test_rmse(y_true, y_pred_stochastic):
    test_rmse = np.sqrt(np.square(y_true - y_pred_stochastic.mean(axis=3)).mean(axis=1))
    ttm, dtm = test_rmse.mean(axis=0)
    tts, dts = test_rmse.std(axis=0)
    return {"temperature": {"mean": ttm, "std": tts}, "density": {"mean": dtm, "std": dts}}


def test_rmse(y_true, y_pred_stochastic):
    return {"per_sample": per_sample_test_rmse(y_true, y_pred_stochastic),
            "mean": mean_test_rmse(y_true, y_pred_stochastic)}


def per_sample_physical_inconsistency(y_pred_stochastic):
    inconsistencies = phyincon(torch.from_numpy(y_pred_stochastic[:, :, 1, :]), tol=1e-5,
                               r_axis=1, d_axis=1).detach().numpy()
    return {"mean": inconsistencies.mean(axis=(0,-1)), "std": inconsistencies.std(axis=(0,-1))}


def mean_physical_inconsistency(y_pred_stochastic):
    inconsistencies = phyincon(torch.from_numpy(y_pred_stochastic[:, :, 1, :].mean(axis=-1)), tol=1e-5, r_axis=1,
                               d_axis=1).detach().numpy()
    return {"mean": inconsistencies.mean(axis=0), "std": inconsistencies.std(axis=0)}


def test_physical_inconsistency(y_pred_stochastic):
    return {"per_sample": per_sample_physical_inconsistency(y_pred_stochastic),
            "mean": mean_physical_inconsistency(y_pred_stochastic)}

def prefix_nested(col: str) -> pl.Expr:
    return pl.col(col).name.prefix_fields(col + ".")


def table_from_results(test_results):
    return (
        pl.from_dicts(test_results)
        .with_columns(prefix_nested("rmse"), prefix_nested("physical_inconsistency"), with_glm=pl.lit(True))
        .unnest("rmse", "physical_inconsistency")
        .with_columns(prefix_nested("rmse.per_sample"), prefix_nested("rmse.mean"), prefix_nested("rmse.their"))
        .unnest("rmse.per_sample", "rmse.mean", "rmse.their")
    )

def table_from_results_glm(test_results_w_glm, test_results_wo_glm):
    tab_w = table_from_results(test_results_w_glm).with_columns(with_glm=pl.lit(True))
    tab_wo = table_from_results(test_results_wo_glm).with_columns(with_glm=pl.lit(False))
    return pl.concat([tab_w, tab_wo])