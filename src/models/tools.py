import torch
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

def descending_fraction(input: torch.Tensor, tol=0):
    """ Fraction of descending terms in the last dimension of the given tensor
    @param input: a tensor
    @param tol: tolerance to consider a term descending
    @return:
    """
    if input.shape[-1] < 2:
        return torch.zeros(input.shape[:-1], device=input.device, dtype=input.dtype)
    deltas = input.diff(dim=-1)
    return torch.mean(deltas < -abs(tol), dtype=deltas.dtype)

def physical_inconsistency(input: torch.Tensor, tol=0):
    return descending_fraction(input, tol=tol)

def physical_consistency(input: torch.Tensor, tol=0):
    return 1 - descending_fraction(input, tol=tol)

def count_parameters(model, all):
    return sum(p.numel() for p in model.parameters() if all or p.requires_grad)
