import torch

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

def monotonic_loss(input: torch.Tensor, target=None, ascending=True, strict=True):
    if input.shape[-1] < 2:
        return torch.zeros(input.shape[:-1], device=input.device, dtype=input.dtype)
    deltas = input.diff(dim=-1)
    if ascending:
        if strict:
            n_transgressions = (deltas <= torch.zeros_like(deltas))
        else:
            n_transgressions = (deltas < torch.zeros_like(deltas))
    else:
        if strict:
            n_transgressions = (deltas >= torch.zeros_like(deltas))
        else:
            n_transgressions = (deltas > torch.zeros_like(deltas))
    return n_transgressions.mean(dtype=input.dtype)
