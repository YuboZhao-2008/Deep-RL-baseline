"""
Minimal PyTorch utilities for Alphagen.

Provides masked mean/std and per-day normalisation functions used by
the correlation utilities and calculators.
"""
from typing import Tuple, Optional
import torch
from torch import Tensor

def masked_mean_std(
    x: Tensor,
    n: Optional[Tensor] = None,
    mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Compute per-row mean and standard deviation of a 2-D tensor with NaN masking.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (days, stocks).
    n : Tensor, optional
        Precomputed count of non-masked elements per row.
    mask : Tensor, optional
        Boolean mask indicating NaN positions; True values are ignored in the
        statistics.  If not provided, mask is derived from ``torch.isnan(x)``.

    Returns
    -------
    mean : Tensor
        Tensor of shape (days,) containing the mean of each row.
    std : Tensor
        Tensor of shape (days,) containing the standard deviation of each row.
    """
    if mask is None:
        mask = torch.isnan(x)
    if n is None:
        n = (~mask).sum(dim=1)
    x = x.clone()
    x[mask] = 0.0
    mean = x.sum(dim=1) / n
    std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()
    return mean, std

def normalize_by_day(value: Tensor) -> Tensor:
    """
    Normalise each row of a 2-D tensor by subtracting its mean and dividing
    by its standard deviation.  NaNs are replaced with zeros.

    Returns a new tensor of the same shape.
    """
    mean, std = masked_mean_std(value)
    value = (value - mean[:, None]) / std[:, None]
    nan_mask = torch.isnan(value)
    value[nan_mask] = 0.0
    return value
