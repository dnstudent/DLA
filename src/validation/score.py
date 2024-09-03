from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
import numpy as np

def windowed_metric(metric):
    def _fn(y_true, y_pred, **kwargs):
        n_outputs = y_pred.shape[-1]
        return metric(y_true.reshape((-1, n_outputs)), y_pred.reshape((-1, n_outputs)), **kwargs)
    return _fn

@windowed_metric
def w_neg_root_mean_squared_error(y_true, y_pred, **kwargs):
    return -root_mean_squared_error(y_true, y_pred, **kwargs)

@windowed_metric
def w_r2_score(y_true, y_pred, **kwargs):
    return r2_score(y_true, y_pred, **kwargs)

def per_sample_test_rmse(y_true, y_pred_stochastic):
    rmses = np.sqrt(np.square(y_true[..., None] - y_pred_stochastic).mean(axis=(0,1)))
    return rmses.mean(axis=-1), rmses.std(axis=-1)