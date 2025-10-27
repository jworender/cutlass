"""
CUTLASS package
---------------

Reusable implementation of the rectified L1 logistic regression workflow
developed in the accompanying research project.
"""

from __future__ import annotations

from ._version import __version__
from .metrics import calculate_youden_j, precision_recall_curve, roc_auc_score
from .model import CutlassClassifier
from .preprocessing import Rectifier, StandardScaler
from .linear_model import CutlassLogisticCV

__all__ = [
    "__version__",
    "CutlassClassifier",
    "CutlassLogisticCV",
    "Rectifier",
    "StandardScaler",
    "calculate_youden_j",
    "precision_recall_curve",
    "roc_auc_score",
]
