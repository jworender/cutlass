"""Minimal pipeline utilities used within the CUTLASS API."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["Pipeline"]


class Pipeline:
    """
    Lightweight analogue of :class:`sklearn.pipeline.Pipeline`.

    Parameters
    ----------
    steps :
        Sequence of (name, estimator/transformer) pairs.
    """

    def __init__(self, steps: Sequence[Tuple[str, object]]) -> None:
        self.steps: List[Tuple[str, object]] = list(steps)
        self.named_steps = {name: step for name, step in self.steps}

    def fit(self, X: np.ndarray, y: Iterable[int]) -> "Pipeline":
        Xt = np.asarray(X, dtype=np.float64)
        target = np.asarray(list(y))
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, target)
            else:
                step.fit(Xt, target)
                if hasattr(step, "transform"):
                    Xt = step.transform(Xt)
            self.named_steps[name] = step
        last_name, last = self.steps[-1]
        last.fit(Xt, target)
        self.named_steps[last_name] = last
        self._fitted_shapes_ = Xt.shape
        return self

    def predict_proba(self, X: np.ndarray):
        Xt = np.asarray(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        _, last = self.steps[-1]
        return last.predict_proba(Xt)

