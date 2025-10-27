"""
High-level estimator that exposes the CUTLASS workflow with a
scikit-learn-inspired API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .linear_model import CutlassLogisticCV
from .preprocessing import Rectifier, StandardScaler

__all__ = ["CutlassClassifier"]


def _ensure_dataframe(
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if feature_names is None:
        raise ValueError("feature_names must be supplied when passing an array for X.")
    return pd.DataFrame(np.asarray(X), columns=list(feature_names))


def _is_binary_pm1(array: np.ndarray) -> bool:
    vals = np.unique(array)
    return set(vals.tolist()).issubset({-1, 1})


@dataclass
class CutlassClassifier:
    """
    End-to-end classifier that (optionally) performs CUTLASS rectification
    followed by an L1-penalised logistic model with cross-validation.

    Parameters mirror the research scripts so the estimator can be used as a
    drop-in replacement inside scikit-learn workflows (fit / predict / score).
    """

    rectify: bool = True
    groups: Optional[Mapping[str, Sequence[str]]] = None
    sdfilter: Optional[float] = 3.0
    snap: float = 0.001
    exclude_features: Sequence[str] = ()
    use_scaler: Optional[bool] = None
    Cs: int | Sequence[float] = 15
    solver: str = "cd"
    cv: int = 3
    tol: float = 1e-4
    max_iter: int = 2000
    random_state: int = 42
    cv_rule: str = "min"
    zero_clamp: float = 0.0
    logic_polish: bool = False
    logic_scale: float = 10.0
    logic_target: Optional[float] = None
    logic_maxk: Optional[int] = None
    logic_rel_tol: float = 0.01
    logic_plot: bool = False
    logic_plot_dir: Optional[str] = None
    logic_Ks_plot: Optional[Sequence[float]] = None
    logic_k_policy: str = "global"
    logic_smooth_k: int = 3
    logic_firstmin_drop: float = 0.05
    logic_firstmin_frac: float = 0.5
    logic_intercept: str = "mean"
    logic_m: Optional[int] = None
    logic_m_frac: Optional[float] = None
    verbose: bool = True

    # ------------------------------------------------------------------ #
    # Fitting / prediction API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[int] | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "CutlassClassifier":
        df = _ensure_dataframe(X, feature_names)
        y_arr = np.asarray(y, dtype=int)
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array of binary labels.")
        if df.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        excludes = set(map(str, self.exclude_features))
        df = df[[col for col in df.columns if col not in excludes]]

        if self.rectify:
            rectifier = Rectifier(
                groups=self.groups,
                sdfilter=self.sdfilter,
                snap=self.snap,
                exclude_features=self.exclude_features,
            )
            X_rect = rectifier.fit_transform(df, y_arr)
            feature_order = rectifier.feature_names_
            self.rectifier_ = rectifier
        else:
            feature_order = list(df.columns)
            X_rect = df.to_numpy(dtype=np.float64, copy=False)
            self.rectifier_ = None

        # Decide on scaling
        if self.use_scaler is None:
            do_scale = not (self.rectify and _is_binary_pm1(X_rect))
        else:
            do_scale = bool(self.use_scaler)

        if do_scale:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_proc = scaler.fit_transform(X_rect)
            self.scaler_ = scaler
        else:
            self.scaler_ = None
            X_proc = np.asarray(X_rect, dtype=np.float64)

        lr = CutlassLogisticCV(
            Cs=self.Cs,
            penalty="l1",
            solver=self.solver,
            scoring="neg_log_loss",
            cv=self.cv,
            n_jobs=-1,
            tol=self.tol,
            max_iter=self.max_iter,
            refit=True,
            random_state=self.random_state,
            verbose=self.verbose,
            cv_rule=self.cv_rule,
            zero_clamp=self.zero_clamp,
            logic_polish=self.logic_polish,
            logic_scale=self.logic_scale,
            logic_target=self.logic_target,
            logic_maxk=self.logic_maxk,
            logic_rel_tol=self.logic_rel_tol,
            logic_plot=self.logic_plot,
            logic_plot_dir=self.logic_plot_dir,
            logic_Ks_plot=self.logic_Ks_plot,
            logic_k_policy=self.logic_k_policy,
            logic_smooth_k=self.logic_smooth_k,
            logic_firstmin_drop=self.logic_firstmin_drop,
            logic_firstmin_frac=self.logic_firstmin_frac,
            logic_intercept=self.logic_intercept,
            logic_m=self.logic_m,
            logic_m_frac=self.logic_m_frac,
        )
        lr.fit(X_proc, y_arr)

        self.classifier_ = lr
        self.feature_names_ = feature_order
        self.classes_ = lr.classes_
        self.shape_ = X_proc.shape
        return self

    # Prediction -------------------------------------------------------- #

    def _prepare_features(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("fit must be called before predict.")

        df = _ensure_dataframe(X, feature_names)
        if self.rectifier_ is not None:
            X_rect = self.rectifier_.transform(df)
        else:
            missing = [c for c in self.feature_names_ if c not in df.columns]
            if missing:
                raise ValueError(f"Input is missing features: {missing}")
            X_rect = df[self.feature_names_].to_numpy(dtype=np.float64, copy=False)

        if self.scaler_ is not None:
            return self.scaler_.transform(X_rect)
        return np.asarray(X_rect, dtype=np.float64)

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        X_proc = self._prepare_features(X, feature_names=feature_names)
        return self.classifier_.predict_proba(X_proc)

    def decision_function(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        proba = self.predict_proba(X, feature_names=feature_names)
        return proba[:, 1] - proba[:, 0]

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        proba = self.predict_proba(X, feature_names=feature_names)
        return (proba[:, 1] >= float(threshold)).astype(int)

    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[int] | np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> float:
        preds = self.predict(X, feature_names=feature_names)
        y_arr = np.asarray(y, dtype=int)
        return float(np.mean(preds == y_arr))

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return {
            "rectify": self.rectify,
            "groups": self.groups,
            "sdfilter": self.sdfilter,
            "snap": self.snap,
            "exclude_features": self.exclude_features,
            "use_scaler": self.use_scaler,
            "Cs": self.Cs,
            "solver": self.solver,
            "cv": self.cv,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "cv_rule": self.cv_rule,
            "zero_clamp": self.zero_clamp,
            "logic_polish": self.logic_polish,
            "logic_scale": self.logic_scale,
            "logic_target": self.logic_target,
            "logic_maxk": self.logic_maxk,
            "logic_rel_tol": self.logic_rel_tol,
            "logic_plot": self.logic_plot,
            "logic_plot_dir": self.logic_plot_dir,
            "logic_Ks_plot": self.logic_Ks_plot,
            "logic_k_policy": self.logic_k_policy,
            "logic_smooth_k": self.logic_smooth_k,
            "logic_firstmin_drop": self.logic_firstmin_drop,
            "logic_firstmin_frac": self.logic_firstmin_frac,
            "logic_intercept": self.logic_intercept,
            "logic_m": self.logic_m,
            "logic_m_frac": self.logic_m_frac,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "CutlassClassifier":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter '{key}'.")
            setattr(self, key, value)
        return self

    @property
    def limits_(self):
        """Expose rectification limits when rectification is enabled."""
        if getattr(self, "rectifier_", None) is None:
            raise AttributeError("limits_ is only available when rectify=True.")
        return self.rectifier_.limits_
