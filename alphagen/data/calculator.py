"""
Alpha calculator classes with caching for performance.

This module defines an abstract calculator interface and a tensor-based
implementation that caches evaluated alpha expressions to avoid
re-evaluating them multiple times.  Correlations are computed using
vectorised operations.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence
import torch
from torch import Tensor

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr

class AlphaCalculator(metaclass=ABCMeta):
    """
    Abstract base class for computing information coefficients (IC) and
    related metrics for alpha expressions.
    """
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        """Calculate the IC between a single alpha and a predefined target."""

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        """Calculate the rank IC between a single alpha and a predefined target."""

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        """Return both IC and rank IC for a single alpha."""
        return self.calc_single_IC_ret(expr), self.calc_single_rIC_ret(expr)

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        """Calculate the IC between two alpha expressions."""

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        """Combine alphas linearly and compute the IC of the combination."""

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        """Combine alphas linearly and compute the rank IC of the combination."""

    @abstractmethod
    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        """Combine alphas linearly and return both IC and rank IC."""

class TensorAlphaCalculator(AlphaCalculator):
    """
    Concrete calculator that evaluates alpha expressions on torch tensors.

    It caches evaluated expressions to reduce redundant computation when
    the same expression is used repeatedly (e.g. mutual IC calculations).
    """
    def __init__(self, target: Optional[Tensor]) -> None:
        # Target is expected to be normalised (days, stocks)
        self._target = target
        # Simple string-keyed cache: {str(expr): evaluated tensor}
        self._eval_cache: dict[str, Tensor] = {}

    @property
    @abstractmethod
    def n_days(self) -> int: ...

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError("A target must be set before calculating non-mutual IC.")
        return self._target

    @abstractmethod
    def evaluate_alpha(self, expr: Expression) -> Tensor:
        """Evaluate an alpha into a ``Tensor`` of shape (days, stocks)."""

    def evaluate_alpha_cached(self, expr: Expression) -> Tensor:
        """
        Evaluate an alpha using a simple cache keyed by its string representation.

        Caching avoids re-computation of the same expression when computing
        mutual ICs or repeated evaluations.
        """
        key = str(expr)
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        value = self.evaluate_alpha(expr)
        self._eval_cache[key] = value
        return value

    def make_ensemble_alpha(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tensor:
        """
        Evaluate a linear combination of alpha expressions into a tensor.
        """
        n = len(exprs)
        # Use cached evaluations to avoid repeated evaluate_alpha calls
        factors = [self.evaluate_alpha_cached(exprs[i]) * weights[i] for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def _IR_from_batch(self, batch: Tensor) -> float:
        mean, std = batch.mean(), batch.std()
        return (mean / std).item()

    def _calc_ICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_pearsonr(value1, value2))

    def _calc_rICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_spearmanr(value1, value2))

    # --- Single alpha metrics ---
    def calc_single_IC_ret(self, expr: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha_cached(expr), self.target)

    def calc_single_IC_ret_daily(self, expr: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha_cached(expr), self.target)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self._calc_rIC(self.evaluate_alpha_cached(expr), self.target)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self.evaluate_alpha_cached(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    # --- Mutual metrics ---
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha_cached(expr1), self.evaluate_alpha_cached(expr2))

    def calc_mutual_IC_daily(self, expr1: Expression, expr2: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha_cached(expr1), self.evaluate_alpha_cached(expr2))

    # --- Pool metrics ---
    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(value, self.target)

    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_pool_all_ret_with_ir(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float, float, float]:
        """
        Returns IC, ICIR, Rank IC, Rank ICIR for a linear combination of alphas.
        """
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            ics = batch_pearsonr(value, target)
            rics = batch_spearmanr(value, target)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std
