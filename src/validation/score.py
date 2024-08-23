from sklearn.metrics import root_mean_squared_error, r2_score

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