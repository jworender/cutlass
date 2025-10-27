"""
Low-level numerical helpers used throughout the CUTLASS package.

The functions in this module are intentionally lightweight so they can be
imported by both the solvers and the higher level estimators without creating
cyclic dependencies.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "_sigmoid",
    "_softplus",
    "_binary_log_loss_from_logits",
    "_soft_threshold",
]


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z: np.ndarray | float) -> np.ndarray | float:
    """Stable computation of log(1 + exp(z))."""
    z = np.asarray(z)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _binary_log_loss_from_logits(y: np.ndarray, z: np.ndarray) -> float:
    """
    Binary cross-entropy given logits.

    Parameters
    ----------
    y :
        Binary labels in {0, 1}.
    z :
        Logits (X w + b).
    """
    return float(np.mean(_softplus(z) - y * z))


def _soft_threshold(w: np.ndarray, thresh: float) -> np.ndarray:
    """Proximal operator for the L1 norm."""
    return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

