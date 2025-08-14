"""
Correlation utilities with improved performance.

This module provides functions to compute batch Pearson and Spearman correlations
between two batched tensors.  The Spearman rank is computed using a vectorised
ranking based on ``torch.argsort`` to avoid Python-level loops.
"""

from typing import Tuple
import torch
from torch import Tensor

from alphagen.utils.pytorch_utils import masked_mean_std

def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan
                     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Replace NaN values in either x or y with a fill value and compute
    a mask of valid positions.  Returns the masked x, masked y, the
    count of non-NaN elements per row, and the mask itself.
    """
    # Copy to avoid modifying original tensors
    x = x.clone()
    y = y.clone()
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask

def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    """
    Compute dense ranks along the last dimension for each row of x.

    This implementation uses ``torch.argsort`` twice to obtain dense
    ranks in ``O(N log N)`` time and avoids explicit Python loops.
    NaN positions are set to zero in the returned rank tensor.
    """
    # argsort along the last dimension returns indices of sorted elements
    sorted_idx = x.argsort(dim=1)
    # argsort of sorted indices gives 0-based ranks
    ranks = sorted_idx.argsort(dim=1).to(dtype=x.dtype)
    # Set ranks at NaN positions to zero
    ranks[nan_mask] = 0
    return ranks

def _batch_pearsonr_given_mask(
    x: Tensor, y: Tensor,
    n: Tensor, mask: Tensor
) -> Tensor:
    """
    Compute Pearson correlation row-wise on masked data.
    """
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std * y_std
    # Avoid division by zero for near-constant rows
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / stdmul
    return corrs

def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute Spearman correlation for each row of x and y.
    """
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)

def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute Pearson correlation for each row of x and y.
    """
    # Use zero fill for missing values
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.))
