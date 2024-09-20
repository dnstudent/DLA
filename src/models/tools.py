import torch
from torch.nn import Parameter
from torchmetrics.functional import mean_squared_error

from src.datasets.tools import density

def reverse_tz_loss(t_hat: torch.Tensor, z_hat: torch.Tensor, z_mean, z_std, t_mean, t_std):
    T_hat = t_hat*t_std + t_mean
    return mean_squared_error(z_hat, (density(T_hat) - z_mean) / z_std)

def monotonic_entity_loss(input: torch.Tensor, target=None, ascending=True):
    """Computes the monotonic loss on input, defined as the sum of squared transgressions of monotonicity
    @param input: a tensor having the monotonic quantity on the last dimension
    @param target: ignored. Used for compatibility purposes
    """
    deltas = input.diff(dim=-1)
    if ascending:
        transgressions = deltas.minimum(torch.zeros_like(deltas))
    else:
        transgressions = deltas.maximum(torch.zeros_like(deltas))
    return transgressions.square().sum(dim=-1)

def descending_fraction(input: torch.Tensor, tol, axis, agg_dims):
    """ Fraction of descending terms in the last dimension of the given tensor
    @param agg_dims:
    @param axis:
    @param input: a tensor
    @param tol: tolerance to consider a term descending
    @return:
    """
    if axis < 0:
        axis = input.dim() + axis
    if input.shape[axis] < 2:
        return torch.zeros(input.shape[:axis] + input.shape[axis+1:], device=input.device, dtype=input.dtype)
    deltas = input.diff(dim=axis)
    return torch.mean(deltas < -abs(tol), dtype=deltas.dtype, dim=agg_dims)

def ordered_fraction(input: torch.Tensor, cmp, tol, r_axis=(0,1,2), d_axis=-2):
    """ Fraction of descending terms in the penultimate dimension of the given tensor
    @return:
    """
    deltas = input.diff(dim=d_axis)
    return cmp(deltas, tol).mean(dtype=deltas.dtype, dim=r_axis)

def physical_inconsistency(input: torch.Tensor, tol=0, r_axis=(0,1,2), d_axis=-2):
    return ordered_fraction(input, torch.less_equal, -abs(tol), r_axis, d_axis)

def physical_consistency(input: torch.Tensor, tol=0, r_axis=(0,1,2), d_axis=-2):
    return ordered_fraction(input, torch.greater, -abs(tol), r_axis, d_axis)

def count_parameters(model, all):
    return sum(p.numel() for p in model.parameters() if all or p.requires_grad)


def make_base_weight(*shape) -> Parameter:
    return Parameter(torch.empty(*shape, dtype=torch.float32))

def get_sequential_linear_params(sequential_model, tag):
    return [param for name, param in sequential_model.named_parameters(recurse=True) if tag in name]

def get_sequential_linear_weights(sequential_model):
    return get_sequential_linear_params(sequential_model, tag='weight')

def get_sequential_linear_biases(sequential_model):
    return get_sequential_linear_params(sequential_model, tag='bias')

