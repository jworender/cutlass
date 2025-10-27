"""
Preprocessing utilities for the CUTLASS package.

This module contains
  • a minimalist ``StandardScaler`` compatible with scikit-learn style APIs, and
  • the ``Rectifier`` transformer that implements the critical-range binarisation
    described in the CUTLASS manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["StandardScaler", "Rectifier"]


class StandardScaler:
    """Minimal replacement for :class:`sklearn.preprocessing.StandardScaler`."""

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        if self.with_mean:
            self.mean_ = np.nanmean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=np.float64)

        if self.with_std:
            var = np.nanvar(X, axis=0)
            scale = np.sqrt(var)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1], dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def _organise_by_prefix(feature_names: Sequence[str]) -> Dict[str, List[str]]:
    """
    Group columns by prefix (characters before the trailing digits).

    This replicates the heuristic used in the experimental scripts to keep
    longitudinal measurements grouped together.
    """
    groups: Dict[str, List[str]] = {}
    for feat in feature_names:
        prefix = str(feat)
        while prefix and prefix[-1].isdigit():
            prefix = prefix[:-1]
        groups.setdefault(prefix, []).append(str(feat))
    for prefix, flist in groups.items():
        numeric = [f for f in flist if f[len(prefix):].isdigit()]
        non_numeric = [f for f in flist if not f[len(prefix):].isdigit()]
        groups[prefix] = sorted(
            numeric,
            key=lambda x: int(x[len(prefix):]) if x[len(prefix):] else 0,
        ) + non_numeric
    return groups


def _flatten_group_order(
    groups: Mapping[str, Sequence[str]],
    present_cols: Sequence[str],
) -> List[str]:
    """Return a feature order that respects the requested group ordering."""
    present = set(map(str, present_cols))
    ordered: List[str] = []
    for _, cols in groups.items():
        for col in cols:
            if col in present:
                ordered.append(col)
    for col in present_cols:
        if col not in ordered:
            ordered.append(str(col))
    return ordered


def _limits_from_training(
    feature_order: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    rmin: np.ndarray,
    rmax: np.ndarray,
) -> Dict[str, Dict[str, tuple[float, float]]]:
    """Attach per-feature critical ranges to the group dictionary."""
    limits: Dict[str, Dict[str, tuple[float, float]]] = {}
    pos = {feat: idx for idx, feat in enumerate(feature_order)}
    for group, cols in groups.items():
        group_limits: Dict[str, tuple[float, float]] = {}
        for col in cols:
            if col in pos:
                i = pos[col]
                group_limits[col] = (float(rmin[i]), float(rmax[i]))
        if group_limits:
            limits[group] = group_limits
    return limits


@dataclass
class Rectifier:
    """
    Transformer that converts real-valued features into {-1, +1} indicators
    based on class-conditional critical ranges.
    """

    groups: Optional[Mapping[str, Sequence[str]]] = None
    sdfilter: Optional[float] = 3.0
    snap: float = 0.001
    exclude_features: Sequence[str] = ()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[int] | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "Rectifier":
        X_df, feature_names = self._normalise_input(X, feature_names)
        y_arr = np.asarray(y, dtype=bool)
        if y_arr.shape[0] != X_df.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        groups = self.groups or _organise_by_prefix(feature_names)
        features = _flatten_group_order(groups, feature_names)

        X_mat = X_df[features].to_numpy(dtype=np.float64, copy=False)
        n, p = X_mat.shape

        if not np.any(y_arr):
            rmin = np.full(p, np.nan, dtype=np.float64)
            rmax = np.full(p, np.nan, dtype=np.float64)
        else:
            X_pos = X_mat[y_arr, :]
            mu = np.nanmean(X_pos, axis=0)
            sd = np.nanstd(X_pos, axis=0, ddof=0)
            if self.sdfilter is not None:
                low = mu - self.sdfilter * sd
                high = mu + self.sdfilter * sd
                mask = (X_pos > low) & (X_pos < high)
                Xpf = np.where(mask, X_pos, np.nan)
            else:
                Xpf = X_pos
            rmin = np.nanmin(Xpf, axis=0)
            rmax = np.nanmax(Xpf, axis=0)
            snap_count = max(1.0, float(self.snap) * float(n))
            lower_counts = np.sum(X_mat < rmin, axis=0)
            upper_counts = np.sum(X_mat > rmax, axis=0)
            rmin = np.where(lower_counts < snap_count, np.nan, rmin)
            rmax = np.where(upper_counts < snap_count, np.nan, rmax)

        self.feature_names_ = list(features)
        self.groups_ = {g: list(cols) for g, cols in groups.items()}
        self.rmin_ = rmin.astype(np.float64)
        self.rmax_ = rmax.astype(np.float64)
        self.limits_ = _limits_from_training(self.feature_names_, self.groups_, self.rmin_, self.rmax_)
        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("Rectifier must be fitted before calling transform().")

        X_df, feature_names = self._normalise_input(X, feature_names)

        missing = [col for col in self.feature_names_ if col not in X_df.columns]
        if missing:
            raise ValueError(f"Input is missing rectifier features: {missing}")

        X_mat = X_df[self.feature_names_].to_numpy(dtype=np.float64, copy=False)

        left_fail = np.zeros_like(X_mat, dtype=bool)
        right_fail = np.zeros_like(X_mat, dtype=bool)
        if np.isfinite(self.rmin_).any():
            left_fail = X_mat < self.rmin_
        if np.isfinite(self.rmax_).any():
            right_fail = X_mat > self.rmax_
        outside = left_fail | right_fail

        vec = np.where(outside, -1, 1).astype(np.int8)
        return vec

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[int] | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        return self.fit(X, y, feature_names=feature_names).transform(X, feature_names=feature_names)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _normalise_input(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: Optional[Sequence[str]],
    ) -> tuple[pd.DataFrame, List[str]]:
        exclude = set(map(str, self.exclude_features))

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            if feature_names is None:
                raise ValueError("feature_names must be provided when X is not a DataFrame.")
            df = pd.DataFrame(np.asarray(X), columns=list(feature_names))
        feature_names = [col for col in df.columns if col not in exclude]
        return df, feature_names

