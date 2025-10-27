#!/usr/bin/env python3
"""
Fast script (v5d, scikit-free, high-fidelity & sparse):
- Minimal drop-ins for Pipeline, StandardScaler, LogisticRegressionCV, and metrics.
- L1 logistic regression via:
    * Coordinate Descent (CD) with Sequential Strong Rules + KKT screening (fast, sparse)
    * or FISTA (accelerated prox-gradient) via --solver fista
- Correct scikit parity:
    * penalty term uses λ = 1/(C * n_samples)
    * Cs grid = logspace(-4, 4, cs) when you pass an integer
- Optional CV selection rule: --cv-rule {min,1se}
- Optional post-fit zero clamp for presentation parity: --zero-clamp

Includes optional logical post-processing ("logical polish") after the final refit:
  - logic_polish: bool, off by default
  - logic_scale: float, magnitude K for active coefficients (default 10.0)
  - logic_target: float or None, early-exit if Youden's J >= this (e.g., 0.97)
  - logic_maxk: int or None, cap the number of top features to test (speed) 
"""
# =============================================================================
# Context (Paper tie-in)
# -----------------------------------------------------------------------------
# This script operationalizes the paper’s workflow:
#   1) Learn per-feature "critical ranges" (CRs) from the positive class and
#      then binarize features to {+1, -1} accordingly (Algorithm 1 in paper).
#      Inside CR → +1; outside → -1. This is implemented by `rectify_fast`.
#   2) Fit an L1-penalized logistic model on the rectified design.
#      Theory: binarization reduces pairwise correlations via the arcsin bound
#      |r| ≤ |ρ| (Lemma 1; Appendix B.1–B.2), improving conditioning of the
#      Gram/correlation matrix and helping the IC of Zhao & Yu hold more often.
#      That improves feature recovery and sparsity (Section 4, Lemmas 1–4).
#   3) Optionally compress the fitted model into a simple logical “top‑k votes”
#      rule with fixed-magnitude weights K and an intercept policy such as
#      m-of-k (logic_polish). This produces rule-like models (Section 7).
# Empirical sections (Goose Bay, HAI, UNICEF) motivate the approach.
# =============================================================================

# --- Limit BLAS/OpenMP threads to avoid oversubscription across processes --- #
import os as os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# ^ Constrains low-level math libraries to single-threaded execution by default.
#   Helpful when we also use Python multiprocessing for CV: avoids oversubscribed
#   CPU cores and non-deterministic slowdowns.

import argparse
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import json

from multiprocessing import shared_memory
# ^ We use shared_memory to broadcast X,y once to worker processes in CV so
#   each fold walks the entire C-path without repeated pickling/copying.

#–––– Configuration ––––#
RESP_COL     = 'INDC'
EXCLUDE_COLS = ['X', 'UIC', 'iDate', 'iDate.x', 'iDate.y', 'class', 'cols', 'year']
# ^ Column conventions: response is a binary label 'INDC'. The EXCLUDE_COLS list
#   removes non-feature columns before modeling.

# ====================== Utilities ======================

def timestamp(msg=None):
    now = datetime.now().isoformat(timespec="seconds")
    print(f"{msg+': ' if msg else ''}{now}")
# ^ Small helper for human-friendly progress logs.

def _sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))
# ^ Numerically stable sigmoid: clipping prevents overflow in exp(±z).

def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
# ^ Stable softplus used to compute log(1+exp(z)) without overflow/underflow.

def _binary_log_loss_from_logits(y, z):
    # y in {0,1}, z = Xw + b
    return np.mean(_softplus(z) - y * z)
# ^ Logistic loss in terms of logits; avoids repeated sigmoid calls.

def _soft_threshold(w, thresh):
    return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)
# ^ Proximal operator for L1 norm: the core shrinkage step in CD/FISTA.

def calculate_youden_j(y_true, y_pred, thresh=0.5):
    """
    Youden's J = sensitivity + specificity - 1
    y_true : array-like of {0,1} or bool
    y_pred : array-like of {0,1} or bool (already thresholded to labels)
    thresh : kept for API parity; ignored if inputs are boolean/0-1
    """
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)

    tp = np.sum( yt &  yp)
    tn = np.sum(~yt & ~yp)
    fp = np.sum(~yt &  yp)
    fn = np.sum( yt & ~yp)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens + spec - 1.0)
# ^ Youden’s J is a threshold-based performance metric highlighted in the paper’s
#   tables/figures to compare transformed vs untransformed fits (e.g., Table 2).
#   It is particularly suitable for imbalanced or cost-sensitive scenarios.

def lasso_intercept(X, y, w):
    """
    Approximate intercept for fixed weights (LASSO-style).
    Useful as an initial guess, though the logical polish uses an exact
    1-D solve for the intercept; this can still be called if desired.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feature_means = np.mean(X, axis=0)
    p = np.clip(np.mean(y), 1e-9, 1 - 1e-9)
    lin = float(np.dot(w, feature_means))
    return float(np.log(p / (1.0 - p)) - lin)
# ^ Provides a quick intercept consistent with mean response. In the logic
#   compression step we solve intercept more carefully (e.g., mean-matching).

# --- Saving / loading (portable) ---------------------------------------------
# The following routines serialize a trained "Pipeline" (scaler + L1-logistic) and
# the rectification limits (critical ranges). This mirrors model deployment needs:
#  • persist parameters,
#  • load later and compute predict_proba on new data.

def save_pipeline_npz(pipe, feature_names, path):
    """Persist a trained Pipeline to a single .npz file (portable)."""
    state = {}
    state["feature_names"] = np.array(feature_names, dtype=object)

    # Steps: scaler (optional) + lr (required)
    for name, step in pipe.steps:
        if isinstance(step, StandardScaler):
            state["scaler.with_mean"] = np.array([step.with_mean], dtype=bool)
            state["scaler.with_std"]  = np.array([step.with_std],  dtype=bool)
            state["scaler.mean_"]     = step.mean_.astype(np.float64)
            state["scaler.scale_"]    = step.scale_.astype(np.float64)
        elif isinstance(step, LogisticRegressionCV):
            lr = step
            state["lr.coef_"]      = lr.coef_.ravel().astype(np.float64)
            state["lr.intercept_"] = lr.intercept_.astype(np.float64)
            state["lr.classes_"]   = lr.classes_.astype(int)
            state["lr.C_"]         = np.array([float(lr.C_)], dtype=np.float64)
            state["lr.Cs_"]        = np.array(lr.Cs_, dtype=np.float64)
            state["lr.solver"]     = np.array([lr.solver], dtype=object)
            state["lr.penalty"]    = np.array([lr.penalty], dtype=object)
            # optional extras for bookkeeping
            state["lr.cv_rule"]    = np.array([getattr(lr, "cv_rule", "min")], dtype=object)
            state["lr.zero_clamp"] = np.array([getattr(lr, "zero_clamp", 0.0)], dtype=np.float64)

    np.savez_compressed(path, **state)

def load_pipeline_npz(path):
    """Load a portable .npz and reconstruct a working Pipeline for predict_proba()."""
    blob = np.load(path, allow_pickle=True)
    feature_names = blob["feature_names"].tolist()

    steps = []
    # scaler (optional)
    if "scaler.mean_" in blob:
        sc = StandardScaler(with_mean=bool(blob["scaler.with_mean"][0]),
                            with_std=bool(blob["scaler.with_std"][0]))
        sc.mean_  = blob["scaler.mean_"].astype(np.float64)
        sc.scale_ = blob["scaler.scale_"].astype(np.float64)
        steps.append(("scaler", sc))

    # logistic
    lr = LogisticRegressionCV(
        Cs=blob["lr.Cs_"].astype(np.float64),
        penalty=str(blob["lr.penalty"][0]),
        solver=str(blob["lr.solver"][0]),
        scoring="neg_log_loss",
        cv=3, n_jobs=-1, tol=1e-4, max_iter=2000, refit=False, random_state=42,
        verbose=False,
        cv_rule=str(blob["lr.cv_rule"][0]) if "lr.cv_rule" in blob.files else "min",
        zero_clamp=float(blob["lr.zero_clamp"][0]) if "lr.zero_clamp" in blob.files else 0.0
    )
    lr.coef_      = blob["lr.coef_"].reshape(1, -1)
    lr.intercept_ = blob["lr.intercept_"].astype(np.float64)
    lr.Cs_        = blob["lr.Cs_"].astype(np.float64)
    lr.C_         = float(blob["lr.C_"][0])
    lr.classes_   = blob["lr.classes_"].astype(int)
    steps.append(("lr", lr))

    pipe = Pipeline(steps)
    return pipe, feature_names

# --- Rectification limits (needed for rectified model deployment) ------------
# Serialization helpers for the CR (critical range) limits. The "limits" record
# how we rectified training data so we can apply the SAME ranges to future data.

def _nan_to_none(x):  # JSON-friendly
    return None if (x is None or (isinstance(x, float) and (x != x))) else x

def _none_to_nan(x):
    return np.nan if x is None else x

def save_limits_json(limits: dict, path: str):
    serial = {g: {f: [_nan_to_none(lo), _nan_to_none(hi)]
                  for f, (lo, hi) in d.items()}
              for g, d in limits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serial, f)

def load_limits_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        serial = json.load(f)
    return {g: {f: (_none_to_nan(lo), _none_to_nan(hi))
                for f, (lo, hi) in d.items()}
            for g, d in serial.items()}

def _rankdata_average(a):
    a = np.asarray(a)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_a = a[order]
    n = a.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks
# ^ Simple tie-aware ranking used to implement AUC via Wilcoxon/Mann–Whitney.

def roc_auc_score(y_true, y_score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=np.float64)
    pos = (y == 1); neg = (y == 0)
    n_pos = int(np.sum(pos)); n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = _rankdata_average(s)
    sum_pos = np.sum(r[pos])
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def precision_recall_curve(y_true, probas_pred):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probas_pred, dtype=np.float64)
    desc = np.argsort(p)[::-1]
    y_sorted = y[desc]; p_sorted = p[desc]
    tps = np.cumsum(y_sorted); fps = np.cumsum(1 - y_sorted)
    P = tps[-1] if tps.size else 0
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / np.maximum(P, 1e-12)
    distinct = np.where(np.diff(p_sorted))[0]
    thresholds = p_sorted[distinct]
    last_ind = distinct
    precision = np.r_[precision[last_ind], precision[-1] if precision.size else 1.0]
    recall = np.r_[recall[last_ind], 1.0 if P > 0 else 0.0]
    return precision, recall, thresholds
# ^ Minimal implementations keep the script scikit-free and reproducible.

# ================== Minimal drop-ins ===================

class StandardScaler:
    # -------------------------------------------------------------------------
    # Minimal version of sklearn.preprocessing.StandardScaler.
    # Only what is needed: mean/variance along columns, with options to disable
    # mean or std. Used on RAW (unrectified) continuous features only; when the
    # design is {±1} after rectification, scaling is unnecessary and disabled.
    # -------------------------------------------------------------------------
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
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

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class Pipeline:
    # -------------------------------------------------------------------------
    # Minimal sklearn.pipeline.Pipeline drop-in: chains scaler -> classifier.
    # We capture fitted steps in `named_steps` and route predict_proba through.
    # -------------------------------------------------------------------------
    def __init__(self, steps: List[Tuple[str, object]]):
        self.steps = list(steps)
        self.named_steps = {name: step for name, step in self.steps}

    def fit(self, X, y):
        Xt = np.asarray(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, None)
            else:
                step.fit(Xt, y)
                if hasattr(step, "transform"):
                    Xt = step.transform(Xt)
            self.named_steps[name] = step
        last_name, last = self.steps[-1]
        last.fit(Xt, y)
        self.named_steps[last_name] = last
        self._fitted_shapes_ = Xt.shape
        return self

    def predict_proba(self, X):
        Xt = np.asarray(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        last_name, last = self.steps[-1]
        return last.predict_proba(Xt)

# ======== Optimizers for L1-regularized Logistic ========
# We offer two solvers:
#  • _CDLogistic: coordinate descent with Sequential Strong Rules and KKT
#    screening. On rectified {±1} data, we can use a faster Hessian diagonal.
#  • _FISTALogistic: accelerated proximal gradient with backtracking.

class _CDLogistic:
    """
    Coordinate Descent with SSR + KKT screening:
    min_{w,b}  (1/n) * sum_i [ log(1+exp(z_i)) - y_i z_i ] + lam * ||w||_1
    where z = Xw + b, y in {0,1}. Intercept is unpenalized.
    """
    def __init__(self, lam=1.0, tol=1e-4, max_iter=2000, verbose=False, kkt_tol=1e-4, all_pm1=False):
        self.lam = float(lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = verbose
        self.kkt_tol = float(kkt_tol)
        self.all_pm1 = bool(all_pm1)
        self.w_ = None
        self.b_ = 0.0
        self.n_iter_ = 0
        # NOTE (theory): When features are ±1-coded (our rectified design),
        # the Hessian diagonal simplifies: H_jj ≈ (sum_i p_i(1-p_i))/n (constant),
        # which reduces per-coordinate work. This is a computational reflection of
        # the paper’s benefit: rectification regularizes the design’s geometry.

    def fit(self, X, y, w0=None, b0=None, lam_prev=None, active_init=None):
        """
        Coordinate Descent (binary logistic + L1). Works with either C- or F-contiguous
        arrays. We initially *forced* Fortran order here but shifted away since CV workers
        may receive C-contiguous views backed by shared memory; insisting on order='F' 
        with copy=False raises a ValueError. If the caller *does* pass an F-contiguous
        matrix, we use it as-is (zero-copy).
        """
        # Accept whatever layout we get (zero-copy if already float64)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    
        n, p = X.shape
        w = np.zeros(p, dtype=np.float64) if w0 is None else w0.astype(np.float64).copy()
    
        # Intercept warm start
        if b0 is None:
            py = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
            b = float(np.log(py / (1 - py)))
        else:
            b = float(b0)
    
        lam = self.lam
        z = X @ w + b
        p_hat = _sigmoid(z)
        wght = p_hat * (1 - p_hat)
    
        all_pm1 = self.all_pm1 or np.all((X == 1) | (X == -1))
        # ^ If X is ±1-coded (as produced by rectification), we can use a constant
        #   diagonal Hessian approximation below (fast). This is common in this script
        #   because rectified features are exactly in {±1}.
    
        # Strong rules threshold on first pass of this lambda
        if lam_prev is None or lam_prev <= 0:
            strong_thr = lam
        else:
            strong_thr = max(0.0, 2 * lam - lam_prev)
        # ^ Sequential Strong Rules (SSR): prune inactive coordinates cheaply
        #   as we move down the regularization path (warm starts).
    
        # Initial gradient and active set
        grad = (X.T @ (p_hat - y)) / max(n, 1)
        active_set = set(np.where(np.abs(grad) >= strong_thr - 1e-12)[0].tolist())
        if active_init is not None:
            active_set.update(int(j) for j in active_init)
        active_set.update(np.where(np.abs(w) > 0)[0].tolist())
    
        for it in range(1, self.max_iter + 1):
            # Intercept Newton step (unpenalized)
            r = p_hat - y
            g_b = float(np.sum(r)) / max(n, 1)
            H_bb = float(np.sum(wght)) / max(n, 1) + 1e-12
            db = - g_b / H_bb
            if db != 0.0:
                b += db
                z += db
                p_hat = _sigmoid(z)
                wght = p_hat * (1 - p_hat)
                r = p_hat - y
    
            # Diagonal Hessian approx
            if all_pm1:
                const = float(np.sum(wght)) / max(n, 1) + 1e-12
                H_diag = np.full(p, const, dtype=np.float64)
            else:
                # einsum handles non-contiguous inputs efficiently
                H_diag = np.einsum('i,ij,ij->j', wght, X, X) / max(n, 1) + 1e-12
    
            max_dw = 0.0
            order = np.array(sorted(list(active_set), key=lambda j: -abs(grad[j])), dtype=int) \
                    if active_set else np.array([], dtype=int)
    
            # Coordinate updates
            for j in order:
                # fresh coordinate gradient
                g_j = float(np.dot(X[:, j], r)) / max(n, 1)
                v = w[j] - g_j / H_diag[j]
                w_new = _soft_threshold(v, lam / H_diag[j])
                delta = w_new - w[j]
                if delta != 0.0:
                    w[j] = w_new
                    z += delta * X[:, j]
                    p_hat = _sigmoid(z)
                    wght = p_hat * (1 - p_hat)
                    r = p_hat - y
                    max_dw = max(max_dw, abs(delta))
    
            # KKT check (adds back violated inactive coords)
            grad = (X.T @ (p_hat - y)) / max(n, 1)
            inactive = np.setdiff1d(np.arange(p), np.array(list(active_set), dtype=int), assume_unique=False)
            viol = np.where(np.abs(grad[inactive]) > lam * (1 + self.kkt_tol))[0] if inactive.size else np.array([], dtype=int)
            if viol.size:
                active_set.update(inactive[viol].tolist())
                if self.verbose:
                    print(f"[CD] Added {viol.size} features from KKT violations.")
            else:
                stat = 0.0
                if active_set:
                    a_idx = np.array(list(active_set), dtype=int)
                    stat = np.max(np.abs(grad[a_idx] + lam * np.sign(w[a_idx])))
                if max_dw <= self.tol and stat <= max(self.tol, 1e-6):
                    self.n_iter_ = it
                    break
    
            self.n_iter_ = it
    
        self.w_ = w
        self.b_ = b
        return self


    def predict_proba(self, X):
        z = X @ self.w_ + self.b_
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

class _FISTALogistic:
    """
    Accelerated Proximal Gradient (FISTA) with backtracking + adaptive restart.
    min (1/n) sum logloss + lam * ||w||_1, intercept unpenalized.
    """
    def __init__(self, lam=1.0, tol=1e-4, max_iter=4000, step=None, verbose=False):
        self.lam = float(lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.step = step
        self.verbose = verbose
        self.w_ = None
        self.b_ = 0.0
        self.n_iter_ = 0

    @staticmethod
    def _estimate_L(X):
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        v = np.random.default_rng(123).standard_normal(p)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(12):
            Xv = X @ v
            v = X.T @ Xv
            nv = np.linalg.norm(v)
            if nv == 0: break
            v /= nv
        smax_sq = np.linalg.norm(X @ v)**2
        return 0.25 * smax_sq / max(n,1) + 0.25  # intercept upper bound added
        # ^ A crude Lipschitz upper bound for the logistic gradient plus an
        #   extra 0.25 for the intercept. Enough for backtracking to work.

    def fit(self, X, y, w0=None, b0=None, **kwargs):
        """
        FISTA for L1-logistic. Accept arbitrary memory layout (C or Fortran).
        This avoids order='F', copy=False conflicts in shared-memory workers.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    
        n, p = X.shape
        w = np.zeros(p) if w0 is None else w0.astype(np.float64).copy()
    
        if b0 is None:
            py = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
            b = float(np.log(py / (1 - py)))
        else:
            b = float(b0)
    
        # Lipschitz estimate; works for any layout
        L = self._estimate_L(X) if self.step is None else 1.0 / self.step
        tstep = 0.9 / L
    
        w_y = w.copy(); b_y = b
        t = 1.0
        prev_obj = np.inf
    
        for it in range(1, self.max_iter + 1):
            z = X @ w_y + b_y
            p_hat = _sigmoid(z)
            grad_w = (X.T @ (p_hat - y)) / max(n, 1)
            grad_b = float(np.sum(p_hat - y)) / max(n, 1)
    
            # Backtracking line search
            found = False
            bt = 0
            while not found and bt < 20:
                w_new = _soft_threshold(w_y - tstep * grad_w, tstep * self.lam)
                b_new = b_y - tstep * grad_b
    
                z_new = X @ w_new + b_new
                obj_new = _binary_log_loss_from_logits(y, z_new) + self.lam * np.sum(np.abs(w_new))
    
                dz_w = w_new - w_y
                dz_b = b_new - b_y
                quad = (_binary_log_loss_from_logits(y, z) +
                        np.dot(grad_w, dz_w) + grad_b * dz_b +
                        (np.linalg.norm(dz_w)**2 + dz_b * dz_b) / (2 * tstep))
                if obj_new <= quad + 1e-12:
                    found = True
                else:
                    tstep *= 0.5
                    bt += 1
    
            # Nesterov + adaptive restart
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            w_acc = w_new + ((t - 1) / t_new) * (w_new - w)
            b_acc = b_new + ((t - 1) / t_new) * (b_new - b)
            if np.dot((w_acc - w_new), (w_new - w)) + (b_acc - b_new) * (b_new - b) > 0:
                w_y, b_y = w_new, b_new
                t = 1.0
            else:
                w_y, b_y = w_acc, b_acc
                t = t_new
    
            dw = np.linalg.norm(w_new - w); db = abs(b_new - b)
            w, b = w_new, b_new
            self.n_iter_ = it
    
            if dw + db <= self.tol * (1.0 + np.linalg.norm(w) + abs(b)):
                break
    
            z_curr = X @ w + b
            obj = _binary_log_loss_from_logits(y, z_curr) + self.lam * np.sum(np.abs(w))
            if obj > prev_obj + 1e-10:
                t = 1.0
                w_y, b_y = w, b
            prev_obj = obj
    
        self.w_ = w
        self.b_ = b
        return self


    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.w_ + self.b_
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

class LogisticRegressionCV:
    """
    Minimal drop-in to replace sklearn.linear_model.LogisticRegressionCV (binary, L1).

    Supported args:
      - Cs (int or array-like): if int, uses np.logspace(-4, 4, Cs)
      - penalty='l1'
      - solver: 'cd' (recommended), or 'fista'; 'saga'/'liblinear' map to 'cd'
      - scoring='neg_log_loss' (only)
      - cv (int folds), tol, max_iter, random_state
      - refit (bool)
      - cv_rule: 'min' (best mean) or '1se' (strongest within one SE of best)
      - zero_clamp: set |w|<=threshold to 0 after final fit (cleanup only)

    Attributes: coef_, intercept_, C_, Cs_, classes_
    
    Includes optional logical post-processing ("logical polish") after the final refit:
      - logic_polish: bool, off by default
      - logic_scale: float, magnitude K for active coefficients (default 10.0)
      - logic_target: float or None, early-exit if Youden's J >= this (e.g., 0.97)
      - logic_maxk: int or None, cap the number of top features to test (speed)
      - logic_rel_tol: float, Adoption tolerance (relative). Adopt logical model if
          J_logical >= J_baseline * (1 - logic_rel_tol). Example: 0.01 allows up to ~1% lower J.
      - logic_plot   : bool, Produce a J vs k plot, and optionally J vs K if a grid is given.
      - logic_plot_dir: Optional[str], If provided and writable, figures are saved there (PNG).
          Otherwise, figures are returned on self.logic_figs_ for the caller.
      - logic_Ks_plot: Optional[Sequence[float]], If provided, also generate 
          J vs K using these K values.
      - logic_k_policy : {'global','premin'}, how to choose k after scanning
            'global'  -> best J over all k (previous behavior)
            'premin'  -> best J for k in [1, k_first_min], where k_first_min is
                         the first “real” valley of the smoothed J_k curve
      - logic_smooth_k : int, odd window size for moving-average smoothing of J_k
      - logic_firstmin_drop : float, declare a valley once the smoothed J has
                         fallen by at least this absolute amount from its
                         running maximum (robust valley detection)
      - logic_firstmin_frac : float in (0,1], fallback: if no valley is found,
                         treat the first ⌈frac·L⌉ points as the search window

    """
    # -------------------------------------------------------------------------
    # Theory tie-in:
    #   • CV across C (equivalently λ = 1/(Cn)) controls sparsity. On rectified
    #     designs, lower cross-correlation helps avoid false inclusions (IC).
    #   • "1se" rule prefers simpler models within one SE of best loss (paper’s
    #     preference for sparse, interpretable solutions).
    #   • logic_polish compresses the L1 model into a rule: fixed-magnitude ±K
    #     on top‑k features + an intercept policy (e.g., m‑of‑k). This reflects
    #     the Boolean framing discussed in the manuscript.
    # -------------------------------------------------------------------------
    def __init__(self, Cs=10, penalty="l1", solver="cd", scoring="neg_log_loss",
                 cv=3, n_jobs=-1, tol=1e-4, max_iter=2000, refit=True,
                 random_state=42, verbose=True,
                 # existing custom knobs
                 cv_rule="min", zero_clamp=0.0,
                 # logical-polish knobs
                 logic_polish=False,
                 logic_scale=10.0,
                 logic_target=None,
                 logic_maxk=None,
                 logic_rel_tol=0.01,         # 1% tolerance default
                 logic_plot=False,
                 logic_plot_dir=None,
                 logic_Ks_plot=None,
                 # k-selection behavior
                 logic_k_policy="",          # "" or "global" or "premin"
                 logic_smooth_k=3,
                 logic_firstmin_drop=0.05,
                 logic_firstmin_frac=0.5,
                 # NEW: intercept policy
                 logic_intercept="mean",     # "mean" | "mofk" | "maxj"
                 logic_m=None,               # for "mofk": integer m (default: m=k)
                 logic_m_frac=None):         # alternatively, fraction in (0,1]; m=ceil(frac*k)
        self.Cs = Cs
        self.penalty = penalty
        self.solver = solver
        self.scoring = scoring
        self.cv = int(cv)
        self.n_jobs = n_jobs
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.refit = bool(refit)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
    
        # extras
        self.cv_rule = str(cv_rule)
        self.zero_clamp = float(zero_clamp)
    
        # logical-polish parameters
        self.logic_polish   = bool(logic_polish)
        self.logic_scale    = float(logic_scale)
        self.logic_target   = None if logic_target is None else float(logic_target)
        self.logic_maxk     = None if logic_maxk is None else int(logic_maxk)
        self.logic_rel_tol  = float(logic_rel_tol)
        self.logic_plot     = bool(logic_plot)
        self.logic_plot_dir = logic_plot_dir
        self.logic_Ks_plot  = None if logic_Ks_plot is None else list(map(float, logic_Ks_plot))
    
        # k-selection
        self.logic_k_policy      = (str(logic_k_policy) or "global").lower()
        self.logic_smooth_k      = int(logic_smooth_k)
        self.logic_firstmin_drop = float(logic_firstmin_drop)
        self.logic_firstmin_frac = float(logic_firstmin_frac)
    
        # NEW: intercept policy
        self.logic_intercept = str(logic_intercept).lower()  # "mean" | "mofk" | "maxj"
        self.logic_m         = None if logic_m is None else int(logic_m)
        self.logic_m_frac    = None if logic_m_frac is None else float(logic_m_frac)
    
        # learned attributes
        self.classes_   = np.array([0, 1], dtype=int)
        self.Cs_        = None
        self.C_         = None
        self.coef_      = None
        self.intercept_ = None
    
        # diagnostics from the logical step (populated if run)
        self.logic_diag_  = {}
        self.logic_figs_  = []


    def _kfold_indices(self, n):
        k = self.cv
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        return np.array_split(idx, k)

    def _make_solver(self, lam, X):
        all_pm1 = np.all((X == 1) | (X == -1))
        s = self.solver.lower()
        if s in ("saga", "liblinear", "cd"):
            return _CDLogistic(lam=lam, tol=self.tol, max_iter=self.max_iter,
                               verbose=self.verbose, kkt_tol=1e-4, all_pm1=all_pm1)
        elif s == "fista":
            return _FISTALogistic(lam=lam, tol=self.tol, max_iter=max(self.max_iter, 4000),
                                  verbose=self.verbose)
        else:
            raise ValueError(f"Unsupported solver '{self.solver}'. Use 'cd' or 'fista'.")

    def _logical_polish(self, X, y, w, b,
                        K=10.0, target=None, rel_tol=0.01, maxk=None,
                        make_plots=False, Ks_plot=None, plot_dir=None,
                        verbose=True):
        """
        Compress final L1 model into a Boolean-style rule on the top-k features.
        NEW: Intercept policy via self.logic_intercept:
            - "mean": mean-matching (existing behavior)
            - "mofk": put 0.5 boundary at m-of-k votes (default m=k -> AND)
            - "maxj": choose b that maximizes Youden's J at threshold 0.5
        Also fixes k off-by-one and restores adoption guard.
        """
        import matplotlib.pyplot as plt
        # ---------------------------------------------------------------------
        # Theory tie-in:
        #  After L1 finds a sparse linear model, we can "quantize" it into a rule:
        #    sign(w_j) → vote direction on feature j (top-|w| features only),
        #    magnitude → constant K,
        #    decision threshold controlled by intercept policy (mean/m-of-k/maxJ).
        #  This mirrors the paper’s Boolean framing: sparse interpretable logic
        #  consistent with threshold-based phenomena (Section 7, discussion).
        # ---------------------------------------------------------------------
    
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        y_bool = (y >= 1)
    
        # --- Baseline J from the original L1 logits (threshold at 0.5 => z>=0) ---
        z0 = X @ w + b
        yhat0 = (z0 >= 0.0)
        J0 = calculate_youden_j(y_bool, yhat0)
    
        # Order features by |w| (drop exact zeros)
        order_all = np.argsort(-np.abs(w))
        mask_nz = np.abs(w[order_all]) > 0
        if not np.any(mask_nz):
            return w, b, float(J0), False, [], {"J0": J0, "k_best": 0}
    
        order = order_all[mask_nz]
        L = len(order)
        if (maxk is not None) and (maxk > 0):
            L = min(L, int(maxk))
    
        # Precompute signed columns and cumulative sums for fast top‑k votes
        signs = np.sign(w[order][:L])
        X_sel = X[:, order[:L]] * signs       # (n, L) -> aligned votes are +1
        cum = np.cumsum(X_sel, axis=1)        # (n, L), vote sum s in {-k,-k+2,...,k}
    
        # ---------- Intercept helpers ----------
        ybar = float(np.mean(y_bool))
    
        def solve_intercept_mean_match(t, b_init=0.0):
            # Newton + bracket to solve mean(sigmoid(t + b)) = ybar
            b_cur = float(b_init)
            for _ in range(8):
                z = t + b_cur
                p_hat = _sigmoid(z)
                g = float(np.mean(p_hat - y_bool))
                h = float(np.mean(p_hat * (1.0 - p_hat)))
                if h < 1e-12:
                    break
                step = g / h
                b_cur -= step
                if abs(step) < 1e-8:
                    return b_cur
            # robust bracket (±40 is fine; sigmoid is already clipped there)
            lo, hi = -40.0, 40.0
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                mval = float(np.mean(_sigmoid(t + mid)) - ybar)
                if mval > 0.0: hi = mid
                else:          lo = mid
            return 0.5 * (lo + hi)
    
        def b_for_mofk(k, m=None):
            """
            Intercept for an 'at-least m-of-k' rule on ±1 votes with weight magnitude K.
            Place the 0.5 boundary halfway between the m and (m-1) vote levels so that
            examples with exactly m aligned votes have a *positive margin* of +K.
              s_m   = 2m - k
              s_m1  = 2(m-1) - k
              s_thr = (s_m + s_m1)/2 = 2m - k - 1
              b     = -K * s_thr
            """
            if m is None:
                m = k  # strict AND by default
            s_thr = 2*int(m) - k - 1
            return -K * s_thr  # no epsilon needed; s_m yields z = +K
    
        def b_for_max_J(t, y_bool):
            # Scan thresholds on t to maximize Youden's J at 0.5
            vals = np.unique(np.sort(t))
            if vals.size == 0:
                return 0.0, calculate_youden_j(y_bool, np.zeros_like(y_bool, dtype=bool))
            mids = 0.5 * (vals[:-1] + vals[1:])
            thr_candidates = np.r_[vals[0] - 1e-9, mids, vals[-1] + 1e-9]
            bestJ, best_thr = -np.inf, float(thr_candidates[0])
            for thr in thr_candidates:
                pred = (t >= thr)          # equivalent to z = t + b >= 0 with b = -thr
                J = calculate_youden_j(y_bool, pred)
                if J > bestJ:
                    bestJ, best_thr = J, float(thr)
            return -best_thr, bestJ
    
        def youden_from_logit(t, b_local):
            pred = (t + b_local >= 0.0)
            return calculate_youden_j(y_bool, pred)
    
        # ----- Scan k = 1..L, record J_k and b_k under the chosen intercept policy -----
        mode = getattr(self, "logic_intercept", "mean")
        m_fixed = getattr(self, "logic_m", None)
        m_frac  = getattr(self, "logic_m_frac", None)
    
        J_k = np.empty(L, dtype=np.float64)
        b_k = np.empty(L, dtype=np.float64)
    
        best_J_overall = float(J0)
        best_k_overall = 0
        best_b_overall = float(b)
    
        for k in range(1, L + 1):
            t_k = K * cum[:, k - 1]
    
            if mode == "mofk":
                if (m_fixed is not None) and (m_fixed >= 1):
                    m_here = int(np.clip(m_fixed, 1, k))
                elif (m_frac is not None) and (0.0 < m_frac <= 1.0):
                    m_here = int(np.ceil(m_frac * k))
                else:
                    m_here = k
                b_here = b_for_mofk(k, m_here)
                J_here = youden_from_logit(t_k, b_here)

            elif mode == "maxj":
                b_here, J_here = b_for_max_J(t_k, y_bool)
    
            else:  # "mean"
                b_init = best_b_overall if best_k_overall > 0 else b
                b_here = solve_intercept_mean_match(t_k, b_init=b_init)
                J_here = youden_from_logit(t_k, b_here)
    
            J_k[k - 1] = J_here
            b_k[k - 1] = b_here
    
            if J_here > best_J_overall:
                best_J_overall, best_k_overall, best_b_overall = J_here, k, b_here
    
            if (target is not None) and (J_here >= float(target)):
                best_J_overall, best_k_overall, best_b_overall = J_here, k, b_here
                break
    
        # ----- Choose k by policy -----
        def _smooth(y, win):
            win = max(3, int(win))
            if win % 2 == 0: win += 1
            if len(y) < win: return y.copy()
            ker = np.ones(win, dtype=np.float64) / win
            ys = np.convolve(y, ker, mode='same')
            half = win // 2
            ys[:half] = y[:half]; ys[-half:] = y[-half:]
            return ys
    
        Js = _smooth(J_k, self.logic_smooth_k)
        # First valley of *smoothed* curve
        runmax = np.maximum.accumulate(Js)
        drop = runmax - Js
        k_first_min = None
        for i in range(1, L - 1):
            if drop[i] >= self.logic_firstmin_drop and Js[i] <= Js[i-1] and Js[i] <= Js[i+1]:
                k_first_min = i + 1  # 1-based
                break
        if k_first_min is None:
            k_first_min = max(1, int(np.ceil(self.logic_firstmin_frac * L)))
    
        if self.logic_k_policy == "premin":
            window_end = max(1, min(k_first_min, L))
            k_cand = int(np.argmax(J_k[:window_end])) + 1
            J_cand = float(J_k[k_cand - 1])
            b_cand = float(b_k[k_cand - 1])
            if verbose:
                print(f"[logical] premin policy: first valley at k≈{k_first_min}, "
                      f"candidate k={k_cand} with J={J_cand:.4f}")
        else:  # "global"
            k_cand = int(np.argmax(J_k)) + 1
            J_cand = float(J_k[k_cand - 1])
            b_cand = float(b_k[k_cand - 1])
            if verbose:
                print(f"[logical] global policy: candidate k={k_cand} with J={J_cand:.4f}")
    
        # ----- Decide adoption (relative tolerance or hard target) -----
        adopt_threshold = (float(target) if target is not None else float(J0) * (1.0 - float(rel_tol)))
        adopted = (k_cand > 0) and (J_cand >= adopt_threshold)
    
        # Build adopted weight vector if accepted
        if adopted:
            w_new = np.zeros_like(w)
            topk_idx = order[:k_cand]                 # <-- fixed off-by-one
            w_new[topk_idx] = K * np.sign(w[topk_idx])
            b_new = float(b_cand)
            J_adopt = float(J_cand)
            if verbose:
                tau = -b_new / float(K)
                print(f"[logical] adopted rule-like model: J={J_adopt:.4f} "
                      f"| K={K:g}, k={k_cand}, 0.5 boundary at vote sum ≥ {tau:.2f}")
        else:
            w_new, b_new, J_adopt = w, b, float(J0)
    
        # ----- Plots & diagnostics (unchanged apart from fixed labels) -----
        figs = []
        diag = {
            "J0": float(J0),
            "J_k": J_k.copy(),
            "J_k_smooth": Js.copy(),
            "b_k": b_k.copy(),
            "order": order.copy(),
            "best_k_overall": int(best_k_overall),
            "best_J_overall": float(best_J_overall),
            "policy": self.logic_k_policy,
            "intercept_policy": mode,
            "k_first_min": int(k_first_min),
            "k_chosen": int(k_cand),
            "J_chosen": float(J_cand),
            "adopted": bool(adopted),
            "K_used": float(K),
            "adopt_threshold": float(adopt_threshold),
        }
    
        def _save_or_collect(fig, fname):
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
                out = os.path.join(plot_dir, fname)
                fig.savefig(out, dpi=150)
                if verbose:
                    print(f"[logical] saved: {out}")
                plt.close(fig)
            else:
                figs.append(fig)
    
        if make_plots:
            ks = np.arange(1, L + 1)
            fig1, ax1 = plt.subplots(figsize=(6.8, 4.2))
            ax1.plot(ks, J_k, marker='o', linewidth=1, label='J (raw)')
            ax1.plot(ks, Js, linewidth=2, alpha=0.6, label='J (smoothed)')
            ax1.axhline(J0, color='k', linestyle='--', linewidth=1.5, label=f"Baseline J0={J0:.3f}")
            if k_first_min is not None and 1 <= k_first_min <= L:
                ax1.axvline(k_first_min, color='gray', linestyle=':', linewidth=1.5, label=f"first min @ k={k_first_min}")
            if adopted:
                ax1.axvline(k_cand, color='C3', linestyle='--', linewidth=1.0, label=f"chosen k={k_cand}")
            ax1.set_xlabel("k (top-|w| features kept)")
            ax1.set_ylabel("Youden's J")
            ax1.set_title(f"Youden's J vs k (logical compression, intercept={mode})")
            ax1.legend(loc='best')
            fig1.tight_layout()
            _save_or_collect(fig1, "logical_J_vs_k.png")
    
            if Ks_plot is not None and len(Ks_plot) > 0:
                bestJ_vs_K = []
                for Ktest in Ks_plot:
                    J_best_thisK = -np.inf
                    b_best_thisK = 0.0
                    for k in range(1, L + 1):
                        t_k = Ktest * cum[:, k - 1]
                        if mode == "mofk":
                            if (self.logic_m is not None) and (self.logic_m >= 1):
                                m_here = int(np.clip(self.logic_m, 1, k))
                            elif (self.logic_m_frac is not None) and (0.0 < self.logic_m_frac <= 1.0):
                                m_here = int(np.ceil(self.logic_m_frac * k))
                            else:
                                m_here = k
                            b_here = b_for_mofk(k, m_here)
                            J_here = youden_from_logit(t_k, b_here)
                        elif mode == "maxj":
                            b_here, J_here = b_for_max_J(t_k, y_bool)
                        else:
                            b_here = solve_intercept_mean_match(t_k, b_init=b_best_thisK)
                            J_here = youden_from_logit(t_k, b_here)
                        if J_here > J_best_thisK:
                            J_best_thisK = J_here
                            b_best_thisK = b_here
                    bestJ_vs_K.append(J_best_thisK)
                bestJ_vs_K = np.array(bestJ_vs_K, dtype=np.float64)
                diag["Ks_plot"] = list(map(float, Ks_plot))
                diag["bestJ_vs_K"] = bestJ_vs_K.copy()
    
                fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
                ax2.plot(Ks_plot, bestJ_vs_K, marker='o', linewidth=1)
                ax2.axhline(J0, color='k', linestyle='--', linewidth=1.5, label=f"Baseline J0={J0:.3f}")
                ax2.set_xlabel("K (magnitude of logical weights)")
                ax2.set_ylabel("Best Youden's J across k")
                ax2.set_title(f"Youden's J vs K (k re-optimized, intercept={mode})")
                ax2.legend(loc='best')
                fig2.tight_layout()
                _save_or_collect(fig2, "logical_J_vs_K.png")
    
        return w_new, b_new, J_adopt, adopted, figs, diag



    def fit(self, X, y):
        """
        Cross-validated L1 logistic with **persistent** process pool:
          - one pool for the whole CV run,
          - one task per fold (each task walks the entire C-path),
          - no double-submission,
          - no process respawn per C,
          - no per-C pickling of warm-starts.
    
        Final refit + logical polish: unchanged.
        """
        import math
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import shared_memory, cpu_count
    
        # Parent-side arrays (we keep X as Fortran for fast column slices in CD)
        X = np.asarray(X, dtype=np.float64, order='F')
        y = np.asarray(y, dtype=int)
        n, p = X.shape
    
        # Build C grid (same semantics)
        if isinstance(self.Cs, (int, np.integer)):
            Cs = np.logspace(-4, 4, int(self.Cs))
        else:
            Cs = np.asarray(self.Cs, dtype=np.float64)
        self.Cs_ = Cs.copy()
    
        if self.penalty.lower() != "l1":
            raise ValueError("Only penalty='l1' is supported.")
        if self.scoring not in ("neg_log_loss",):
            raise ValueError("Only scoring='neg_log_loss' is supported.")
    
        folds = self._kfold_indices(n)  # list of validation indices
        all_pm1 = np.all((X == 1) | (X == -1))
    
        # Path order: large λ→small λ (i.e., small C→large C)
        lam_full = 1.0 / (np.maximum(Cs, 1e-12) * float(n))
        order = np.argsort(lam_full)[::-1]
        Cs_path = Cs[order]
    
        # Hybrid policy: FISTA for CV (fast), CD for final refit
        use_hybrid = (self.solver.lower() == "hybrid")
        solver_name = ("fista" if use_hybrid else self.solver.lower())
    
        # Share X, y ONCE across the whole pool
        shmX = shared_memory.SharedMemory(create=True, size=X.nbytes)
        X_sh = np.ndarray(X.shape, dtype=X.dtype, buffer=shmX.buf, order='F')
        X_sh[...] = X  # one-time copy
    
        shmy = shared_memory.SharedMemory(create=True, size=y.nbytes)
        y_sh = np.ndarray(y.shape, dtype=y.dtype, buffer=shmy.buf)
        y_sh[...] = y  # one-time copy
    
        # Decide pool size
        max_workers = self.n_jobs
        if max_workers in (None, -1):
            max_workers = min(self.cv, max(1, cpu_count() - 1))
        # ^ Use at most (cv) workers to avoid idle processes; leave 1 core free.
    
        try:
            # Prepare arguments: **one task per fold**
            args_list = []
            for f, val_idx in enumerate(folds):
                args_list.append((
                    shmX.name, X.shape, X.dtype.str, 'F',
                    shmy.name, y.shape, y.dtype.str,
                    val_idx.astype(np.int64, copy=False),
                    Cs_path, solver_name, float(self.tol), int(self.max_iter), bool(all_pm1)
                ))
    
            # Launch ONCE; each worker walks the entire path for its fold
            if self.cv == 1 or max_workers == 1:
                # trivial sequential fallback (no pool)
                fold_losses = [ _fold_path_worker_shm(a) for a in args_list ]
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    fold_losses = list(ex.map(_fold_path_worker_shm, args_list))
    
            # Aggregate CV stats across folds for each C on the path
            fold_losses = np.vstack(fold_losses)              # (cv, nC)
            mean_losses = np.mean(fold_losses, axis=0)        # (nC,)
            if self.cv > 1:
                se_losses = np.std(fold_losses, axis=0, ddof=1) / math.sqrt(self.cv)
            else:
                se_losses = np.zeros_like(mean_losses)
    
            # Emit a compact progress log (optional)
            if self.verbose:
                for i, Ci in enumerate(Cs_path):
                    lam_dbg = 1.0 / (float(Ci) * float(n))
                    print(f"[CV|parallel] C={Ci:.4g} (lam≈{lam_dbg:.3e}) -> "
                          f"mean={mean_losses[i]:.6f} ± {se_losses[i]:.6f}")
    
            # Pick C by rule
            if self.cv_rule.lower() == "1se":
                j_best = int(np.argmin(mean_losses))
                thr = mean_losses[j_best] + se_losses[j_best]
                candidates = np.where(mean_losses <= thr)[0]
                j_pick = int(np.min(candidates)) if candidates.size else j_best
                chosen_C = float(Cs_path[j_pick])
            else:
                chosen_C = float(Cs_path[int(np.argmin(mean_losses))])
    
            self.C_ = chosen_C
    
            # ----- Final refit on ALL data -----
            lam_final = 1.0 / (float(self.C_) * float(n))
            if use_hybrid:
                # CD for the final refit to get crisp sparsity
                all_pm1_full = np.all((X == 1) | (X == -1))
                solver = _CDLogistic(lam=lam_final, tol=self.tol, max_iter=self.max_iter,
                                     verbose=self.verbose, kkt_tol=1e-4, all_pm1=all_pm1_full)
                solver.fit(X, y, w0=None, b0=None, lam_prev=None, active_init=None)
            else:
                solver = self._make_solver(lam_final, X)
                if isinstance(solver, _FISTALogistic):
                    solver.fit(X, y, w0=None, b0=None)
                else:
                    solver.fit(X, y, w0=None, b0=None, lam_prev=None, active_init=None)
    
            w_final = solver.w_.copy()
            b_final = float(solver.b_)
    
            # Optional presentation clamp
            if self.zero_clamp > 0.0:
                w_final[np.abs(w_final) <= self.zero_clamp] = 0.0
    
            # Optional logical polish (unchanged)
            if self.logic_polish:
                w_new, b_new, j_new, adopted, figs, diag = self._logical_polish(
                    X=X, y=y, w=w_final, b=b_final,
                    K=self.logic_scale,
                    target=self.logic_target,
                    rel_tol=self.logic_rel_tol,
                    maxk=self.logic_maxk,
                    make_plots=self.logic_plot,
                    Ks_plot=self.logic_Ks_plot,
                    plot_dir=self.logic_plot_dir,
                    verbose=self.verbose
                )
                self.logic_figs_ = figs
                self.logic_diag_ = diag
                if adopted and self.verbose:
                    print(f"[logical] adopted rule-like model: J={j_new:.4f}")
                if adopted:
                    w_final, b_final = w_new, b_new
    
            self.coef_ = w_final.reshape(1, -1)
            self.intercept_ = np.array([b_final], dtype=np.float64)
            return self
    
        finally:
            # Clean shared memory segments
            try:
                X_sh = None; y_sh = None
                shmX.close(); shmy.close()
                shmX.unlink(); shmy.unlink()
            except Exception:
                pass


    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.coef_.ravel() + float(self.intercept_[0])
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

# ================== Data prep & plotting ==================

def organize(df: pd.DataFrame) -> Dict[str, List[str]]:
    # -------------------------------------------------------------------------
    # Longitudinal column grouping:
    #   If feature names encode variable and lag (e.g., "V6TM4"), this groups
    #   contiguous lags per variable. That ordering is helpful later for plots
    #   and to create consistent CR limits per real-world variable+lag.
    # -------------------------------------------------------------------------
    feats = [c for c in df.columns if c not in EXCLUDE_COLS and c != RESP_COL]
    groups: Dict[str, List[str]] = {}
    for feat in feats:
        prefix = re.sub(r'\d+$', '', feat)
        groups.setdefault(prefix, []).append(feat)
    for prefix, flist in groups.items():
        groups[prefix] = sorted(
            [f for f in flist if re.search(r'(\d+)$', f)],
            key=lambda x: int(re.search(r'(\d+)$', x).group(1))
        ) + [f for f in flist if not re.search(r'(\d+)$', f)]
    return groups

from multiprocessing import shared_memory

from multiprocessing import shared_memory
# ^ Duplicate import is harmless; included to keep the original file intact.

def _fold_path_worker_shm(args):
    """
    Run ONE fold across the ENTIRE C-path, warm-starting within the worker.

    Returns
    -------
    val_losses : 1D np.ndarray of length len(Cs_path) with validation log-loss
                 for this fold at each C in the provided order.
    """
    (shm_name_X, shape_X, dtype_X, order_X,
     shm_name_y, shape_y, dtype_y,
     val_idx, Cs_path, solver_name, tol, max_iter, all_pm1) = args

    # Attach to shared arrays (zero-copy)
    shmX = shared_memory.SharedMemory(name=shm_name_X)
    X = np.ndarray(shape_X, dtype=np.dtype(dtype_X), buffer=shmX.buf, order=order_X)
    shmy = shared_memory.SharedMemory(name=shm_name_y)
    y = np.ndarray(shape_y, dtype=np.dtype(dtype_y), buffer=shmy.buf)

    try:
        n, p = X.shape
        # Build train/validation views for this fold (one time)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        Xtr, ytr = X[mask, :], y[mask]
        Xva, yva = X[val_idx, :], y[val_idx]
        ntr = Xtr.shape[0]

        # Warm-start state kept inside the worker across the C-path
        w_ws = np.zeros(p, dtype=np.float64)
        b_ws = None
        active_ws = np.array([], dtype=np.int32)

        val_losses = np.empty(len(Cs_path), dtype=np.float64)
        prev_C = None

        for t, Ci in enumerate(Cs_path):
            lam_ci   = 1.0 / (float(Ci) * float(ntr))
            lam_prev = (1.0 / (float(prev_C) * float(ntr))) if prev_C is not None else None

            if solver_name in ("saga", "liblinear", "cd"):
                solver = _CDLogistic(lam=lam_ci, tol=tol, max_iter=max_iter,
                                     verbose=False, kkt_tol=1e-4, all_pm1=all_pm1)
                solver.fit(Xtr, ytr, w0=w_ws, b0=b_ws, lam_prev=lam_prev, active_init=active_ws)
            else:
                solver = _FISTALogistic(lam=lam_ci, tol=tol, max_iter=max(max_iter, 4000), verbose=False)
                solver.fit(Xtr, ytr, w0=w_ws, b0=b_ws)

            # Validation loss at this C
            z_val = Xva @ solver.w_ + solver.b_
            val_losses[t] = _binary_log_loss_from_logits(yva.astype(float), z_val)

            # Warm-start for the next C
            w_ws = solver.w_
            b_ws = solver.b_
            active_ws = np.where(np.abs(w_ws) > 0)[0].astype(np.int32)
            prev_C = Ci

        return val_losses
    finally:
        # Detach from shared memory (the parent unlinks)
        shmX.close()
        shmy.close()



def _flatten_group_order(groups: Dict[str, List[str]], present_cols: List[str]) -> List[str]:
    ordered = []
    present = set(present_cols)
    for g in sorted(groups.keys()):
        for f in groups[g]:
            if f in present:
                ordered.append(f)
    remaining = [c for c in present_cols if c not in set(ordered) and c != RESP_COL and c not in EXCLUDE_COLS]
    return ordered + remaining

def _limits_from_training(groups: Dict[str, List[str]], features: List[str], rmin: np.ndarray, rmax: np.ndarray) -> Dict[str, Dict[str, Tuple[float, float]]]:
    lims: Dict[str, Dict[str, Tuple[float, float]]] = {g: {} for g in groups}
    f2i = {f:i for i,f in enumerate(features)}
    for g, fl in groups.items():
        for f in fl:
            if f in f2i:
                i = f2i[f]
                lims[g][f] = (float(rmin[i]), float(rmax[i]))
    return lims

def rectify_fast(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    sdfilter: Optional[float] = 3.0,
    snap: float = 0.001,
):
    # -------------------------------------------------------------------------
    # Core of Algorithm 1 (journal paper): compute feature-specific CRs from
    # positives and encode each feature as +1 if inside CR, -1 otherwise. This
    # yields an X in {±1}^p that:
    #   • reduces pairwise correlations via r ≈ (2/π)arcsin(ρ) ⇒ |r| ≤ |ρ|
    #     (Lemma 1, Appendix B.1–B.2), improving conditioning.
    #   • lowers cross-covariance between irrelevant and relevant features, making
    #     the IC more likely to hold (Lemma 3 → Lemma 4/Theorem sketch).
    # Implementation details:
    #   • sdfilter: removes outliers from positive class when defining CR bounds.
    #   • snap: require at least snap*n examples to fall below/above bounds to
    #           keep a finite bound; otherwise set bound to NaN (unbounded).
    #   • `limits` lets us apply training CRs to test data (deployment parity).
    # -------------------------------------------------------------------------
    feat_candidates = [c for c in df.columns if c not in EXCLUDE_COLS and c != RESP_COL]
    features = _flatten_group_order(groups, feat_candidates)

    X = df[features].to_numpy(dtype=np.float64, copy=False)
    y = df[RESP_COL].to_numpy(dtype=bool, copy=False)

    n, p = X.shape
    if limits is None:
        if not np.any(y):
            rmin = np.full(p, np.nan, dtype=np.float64)
            rmax = np.full(p, np.nan, dtype=np.float64)
        else:
            X_pos = X[y, :]
            mu = np.nanmean(X_pos, axis=0)
            sd = np.nanstd(X_pos, axis=0, ddof=0)
            if sdfilter is not None:
                low = mu - sdfilter * sd
                high = mu + sdfilter * sd
                mask = (X_pos > low) & (X_pos < high)
                Xpf = np.where(mask, X_pos, np.nan)
            else:
                Xpf = X_pos
            rmin = np.nanmin(Xpf, axis=0)
            rmax = np.nanmax(Xpf, axis=0)
            snap_count = snap * n
            lower_counts = np.sum(X < rmin, axis=0)
            upper_counts = np.sum(X > rmax, axis=0)
            rmin = np.where(lower_counts < snap_count, np.nan, rmin)
            rmax = np.where(upper_counts < snap_count, np.nan, rmax)
        limits_dict = _limits_from_training(groups, features, rmin, rmax)
    else:
        rmin = np.full(p, np.nan, dtype=np.float64)
        rmax = np.full(p, np.nan, dtype=np.float64)
        for g, fl in groups.items():
            for f in fl:
                if f in df.columns and f in features:
                    i = features.index(f)
                    try:
                        rmin[i], rmax[i] = limits[g][f]
                    except Exception:
                        pass
        limits_dict = limits

    left_fail  = (X < rmin) if np.isfinite(rmin).any() else np.zeros_like(X, dtype=bool)
    right_fail = (X > rmax) if np.isfinite(rmax).any() else np.zeros_like(X, dtype=bool)
    outside = left_fail | right_fail

    vec = np.where(outside, -1, 1).astype(np.int8)
    dnew = pd.DataFrame(vec, columns=features, index=df.index)
    dnew[RESP_COL] = df[RESP_COL]
    return dnew, limits_dict
# NOTE: We produce ±1 rather than {0,1}. ±1 coding centers features and supports
# a symmetric linear decision surface (z = Xw + b). It also simplifies the Hessian
# approximation in CD on rectified data (see _CDLogistic), speeding convergence.

def _is_binary_pm1(dfX: pd.DataFrame) -> bool:
    cols = dfX.columns[:min(len(dfX.columns), 10)]
    vals = np.unique(dfX[cols].to_numpy().ravel())
    return set(vals.tolist()).issubset({-1, 1})

def build_model(
    X: pd.DataFrame,
    y_col: str = RESP_COL,
    cv: int = 3,
    cs: int = 15,
    use_scaler: Optional[bool] = None,
    solver: str = "saga",
    tol: float = 1e-4,
    max_iter: int = 2000,
    random_state: int = 42,
    cv_rule: str = "min",
    zero_clamp: float = 0.0,
    # logical-polish (including new policy)
    logic_polish=True,
    logic_scale=10.0,
    logic_target=None,
    logic_maxk=150,
    logic_rel_tol=0.2,
    logic_plot=True,
    logic_plot_dir="./figs/",
    logic_Ks_plot=None,
    logic_k_policy="global",     # "global" or "premin"
    logic_smooth_k=7,
    logic_firstmin_drop=0.05,
    logic_firstmin_frac=0.5,
    # NEW: intercept policy
    logic_intercept="mofk",      # "mean" | "mofk" | "maxj"
    logic_m=None,                # for "mofk": integer m (default: m=k)
    logic_m_frac=None,           # or fraction in (0,1]; m=ceil(frac*k)
    verbose=True) -> 'Pipeline':

    # -------------------------------------------------------------------------
    # This builds a 2-step pipeline:
    #   1) Optional StandardScaler (for RAW data only)
    #   2) L1-Logistic with CV across C values. If `use_scaler` is None, we infer:
    #        - If already ±1-coded (rectified), no scaling.
    #        - Otherwise, scale continuous data (variance stabilization).
    #   `logic_polish=True` on rectified data to produce a rule-like compressed
    #   model (e.g., m-of-k) favored by the paper for interpretability.
    # -------------------------------------------------------------------------
    Xonly = X.drop(columns=[y_col])
    y = X[y_col].astype(int).to_numpy()
    is_bin = _is_binary_pm1(Xonly) if use_scaler is None else False
    do_scale = (use_scaler if use_scaler is not None else not is_bin)

    Cs = np.logspace(-4, 4, cs)
    lr_cv = LogisticRegressionCV(
        Cs=Cs, penalty="l1", solver=solver, scoring="neg_log_loss",
        cv=cv, n_jobs=-1, tol=tol, max_iter=max_iter, refit=True,
        random_state=random_state, verbose=verbose,
        cv_rule=cv_rule, zero_clamp=zero_clamp,
        logic_polish=logic_polish, logic_scale=logic_scale,
        logic_target=logic_target, logic_maxk=logic_maxk,
        logic_rel_tol=logic_rel_tol, logic_plot=logic_plot,
        logic_plot_dir=logic_plot_dir, logic_Ks_plot=logic_Ks_plot,
        logic_k_policy=logic_k_policy,
        logic_smooth_k=logic_smooth_k,
        logic_firstmin_drop=logic_firstmin_drop,
        logic_firstmin_frac=logic_firstmin_frac,
        # NEW knobs exposed:
        logic_intercept=logic_intercept,
        logic_m=logic_m,
        logic_m_frac=logic_m_frac,
    )
    steps = []
    if do_scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("lr", lr_cv))
    pipe = Pipeline(steps)
    pipe.fit(Xonly.to_numpy(dtype=np.float64, copy=False), y)
    return pipe

def evaluate(pipe: 'Pipeline', X: pd.DataFrame, y_col: str = RESP_COL):
    # -------------------------------------------------------------------------
    # Convenience function to compute predicted probabilities, AUC, and an F1
    # curve across thresholds (for plotting/selection in main()).
    # -------------------------------------------------------------------------
    Xonly = X.drop(columns=[y_col])
    y = X[y_col].astype(int).to_numpy()
    prob = pipe.predict_proba(Xonly.to_numpy(dtype=np.float64, copy=False))[:, 1]
    auc = roc_auc_score(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    fscore = 2 * rec * prec / (rec + prec + 1e-9)
    return prob, y.astype(bool), auc, fscore

def plot_sorted(prob, y_bool, title, h=0.5):
    import numpy as np, matplotlib.pyplot as plt
    # Primary key: prob (ascending).  Tie‑break: y (False/negatives first).
    order = np.lexsort((y_bool.astype(int), prob))

    fig, ax = plt.subplots()
    ax.scatter(np.arange(prob.size), prob[order],
               c=y_bool[order], s=6, alpha=0.8, rasterized=True)
    ax.axhline(h, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sorted Examples (tie-broken)")
    ax.set_ylabel("Model Response")
    fig.tight_layout()
    return fig


def plot_sorted_tiebreak(prob, y_bool, title, h=0.5):
    import numpy as np, matplotlib.pyplot as plt
    # Sort by prob; within exact ties, put negatives first
    # lexsort sorts by the last key first; give it (tiebreak, primary)
    order = np.lexsort((~y_bool, prob))   # ascending prob; within ties: False(neg) < True(pos)
    x = np.arange(prob.size)
    fig, ax = plt.subplots()
    ax.scatter(x, prob[order], c=y_bool[order], s=6, alpha=0.8, rasterized=True)
    ax.axhline(h, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sorted Examples (tie-broken: negatives first)")
    ax.set_ylabel("Model Response")
    fig.tight_layout()
    return fig

def _natural_vtm_key(name: str):
    """
    Parse names like 'V5TM10' -> (5, 10, 'V5TM10') for a natural sort by
    variable index then lag. Unparsable names go to the end.
    """
    m = re.match(r'^V(\d+)TM(\d+)$', name)
    if m:
        return (int(m.group(1)), int(m.group(2)), name)
    return (10**9, 10**9, name)

def coefficient_series_only(pipe: 'Pipeline', feature_names: List[str], *,
                            sort_for_plot: bool = False) -> pd.Series:
    """
    Return coefficients as a Series indexed by feature_names.
    If sort_for_plot=True, return a *reordered copy* that is easier to read:
    sorted by V then TM (this does not affect the fitted model, only the plot).
    """
    lr = pipe.named_steps["lr"]
    coef = lr.coef_.ravel()
    if coef.shape[0] != len(feature_names):
        raise ValueError(f"coef length {coef.shape[0]} != feature_names length {len(feature_names)}")
    s = pd.Series(coef, index=feature_names)
    if sort_for_plot:
        s = s.sort_index(key=lambda idx: [ _natural_vtm_key(n) for n in idx ])
    return s


def plot_coeff_bars(beta: pd.Series, title: str, highlights: Optional[List[str]] = None,
                    annotate: bool = True, fontsize: int = 9, exact_match: bool = True):
    """
    Bar plot of coefficients.
    - If 'beta' was produced with sort_for_plot=True, lags march left->right for each V.
    - Label matching is *exact* by default to avoid 'V9TM1' also matching 'V9TM10'.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(num=title)
    ax = fig.gca()

    x = np.arange(len(beta))
    ax.bar(x, beta.values, linewidth=1)

    m = np.nanmax(np.abs(beta.values)) if len(beta) else 1.0
    ylim = max(m * 1.3, 0.5)
    ax.set_ylim(-0.1*ylim, ylim)

    # Optional exact labels
    if highlights:
        if exact_match:
            # compile exact patterns ^name$
            rx_list = [re.compile(rf'^{re.escape(pat)}$') for pat in highlights]
            def is_hit(name):  # exact match against any pattern
                return any(rx.match(name) for rx in rx_list)
        else:
            # fallback: substring
            def is_hit(name):
                return any(pat in name for pat in highlights)

        for i, name in enumerate(beta.index):
            if is_hit(name):
                val = beta.values[i]
                offset = 0.04 * ylim
                y = val + (offset if val >= 0 else -offset)
                va = 'bottom' if val >= 0 else 'top'
                ax.text(i, y, name, rotation=90, ha='center', va=va, fontsize=fontsize)

    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig

def assert_alignment(pipe: 'Pipeline', X_df: pd.DataFrame, eps: float = 1e-12) -> float:
    """
    Verifies that the weights line up with columns used for prediction.
    Returns max absolute difference between manual logits and pipeline logits.
    """
    Xn = X_df.to_numpy(dtype=np.float64, copy=False)
    lr = pipe.named_steps["lr"]
    w  = lr.coef_.ravel()
    b  = float(lr.intercept_[0])

    # manual
    z_manual = Xn @ w + b

    # pipeline
    p = pipe.predict_proba(Xn)[:, 1]
    p = np.clip(p, eps, 1 - eps)
    z_pipe = np.log(p / (1 - p))

    diff = float(np.max(np.abs(z_manual - z_pipe)))
    print(f"[alignment] max |Δz| = {diff:.3e}")
    return diff

def maybe_show_or_save(fig, title: str, save_dir: Optional[str], show_each: bool):
    import matplotlib.pyplot as plt
    backend = matplotlib.get_backend().lower()
    interactive = any(k in backend for k in ('qt', 'tk', 'wx', 'gtk', 'macosx', 'nbagg', 'ipympl'))
    if save_dir:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', title).strip('_')
        out = os.path.join(save_dir, f"{safe}.png")
        fig.savefig(out, dpi=150)
        print(f"[saved] {out}")
        plt.close(fig)
    else:
        if interactive:
            if show_each:
                try:
                    fig.canvas.manager.set_window_title(title)
                except Exception:
                    pass
                plt.show()
            else:
                fig.show()
        else:
            auto_dir = os.path.abspath("./figs")
            os.makedirs(auto_dir, exist_ok=True)
            safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', title).strip('_')
            out = os.path.join(auto_dir, f"{safe}.png")
            fig.savefig(out, dpi=150)
            print(f"[non-interactive backend: saved] {out}")
            plt.close(fig)

def parse_highlights_arg(arg: str) -> List[str]:
    if not arg:
        return []
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return items


LABELS = ["V5TM10","V6TM6","V9TM1","V10TM0","V15TM5","V25TM3","V30TM7"]
# ^ Default labels used to annotate coefficients in rectified plots (matches
#   the simulated-case ground truth used in the paper for all data sets).

def main():
    # -------------------------------------------------------------------------
    # End-to-end workflow:
    #   1) Load train/test CSVs.
    #   2) Group columns (longitudinal ordering).
    #   3) Rectify (Algorithm 1) on train to compute CRs; apply SAME CRs to test.
    #   4) Train rectified model (no scaling, enable logic_polish for rule-like).
    #   5) Train raw model with scaling (to contrast, typically less sparse).
    #   6) Evaluate, plot curves and coefficient bars, optionally export artifacts.
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".", help="Directory containing case_1_train_data.csv and case_1_test_data.csv")
    ap.add_argument("--cv", type=int, default=3, help="n-fold cross validation")
    ap.add_argument("--cs", type=int, default=25, help="number of C values to try (log-spaced)")
    ap.add_argument("--sdfilter", type=float, default=3.0, help="sdfilter used in rectification")
    ap.add_argument("--snap", type=float, default=0.001, help="snap threshold used in rectification")
    ap.add_argument("--tol", type=float, default=1e-5, help="solver tolerance")
    ap.add_argument("--save-dir", type=str, default="", help="If set, save all figures to this directory")
    ap.add_argument("--show-each", action="store_true", help="Call plt.show() after EACH figure (prevents only-last-figure issue)")
    ap.add_argument("--no-plots", action="store_true", help="skip plotting")
    ap.add_argument("--backend", type=str, default="", help="Matplotlib backend (e.g., TkAgg, QtAgg). If empty, use default.")
    ap.add_argument("--solver", type=str, default="hybrid", help="Solver: cd|saga|liblinear (CD)|FISTA or hybrid (FISTA for CV, CD for final)")
    ap.add_argument("--cv-rule", type=str, default="1se", choices=["min","1se"], help="CV model selection rule")
    ap.add_argument("--zero-clamp", type=float, default=1e-3, help="Set |w|<=zero_clamp to exactly 0 after refit (presentation only)")
    ap.add_argument("--highlights", type=str, default="", help="Comma-separated feature suffixes to label (e.g., 'V5TM10,V6TM6,...)'")
    ap.add_argument("--export-dir", type=str, default="./", help="If set, save models (and rectification limits) here")
    args = ap.parse_args()

    if args.backend:
        try:
            matplotlib.use(args.backend, force=True)
            print(f"[matplotlib] Using backend: {args.backend}")
        except Exception as e:
            print(f"[matplotlib] Could not set backend to {args.backend}: {e}")

    save_dir = args.save_dir if args.save_dir else None
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    timestamp("Start")

    tr_path = os.path.join(args.data_dir, "train_data.csv")
    te_path = os.path.join(args.data_dir, "test_data.csv")
    dset_train = pd.read_csv(tr_path, true_values=["TRUE", "True", "T", "1"], false_values=["FALSE", "False", "F", "0"])
    dset_test  = pd.read_csv(te_path, true_values=["TRUE", "True", "T", "1"], false_values=["FALSE", "False", "F", "0"])

    print("Detecting groups...")
    groups = organize(dset_train)
    print("Done.")

    print("Rectify (vectorized) start"); timestamp()
    rt_train, limits = rectify_fast(dset_train, groups, limits=None,   sdfilter=args.sdfilter, snap=args.snap)
    rt_test,  _      = rectify_fast(dset_test,  groups, limits=limits, sdfilter=args.sdfilter, snap=args.snap)
    print("Rectify stop"); timestamp()
    # ^ Train-time CRs (limits) are re-used on test data to ensure the *same*
    #   binarization mapping is applied—this mirrors deployment.

    print("Fit on rectified (skip scaling)"); timestamp()
    model_rect = build_model(
        rt_train, cv=args.cv, cs=args.cs, use_scaler=False, tol=args.tol,
        solver=args.solver, cv_rule=args.cv_rule, zero_clamp=args.zero_clamp,
        # --- turn plots on and save them under ./figs (or leave dir None to display) ---
        logic_polish=True, logic_plot=True, logic_plot_dir="./figs/",
        # Optional knobs you may want to expose on CLI later:
        logic_scale=10.0, logic_target=None, logic_maxk=150, logic_rel_tol=0.01,
        verbose=True
    )
    # ^ On rectified data we enable logic_polish to yield an interpretable
    #   rule-like model (top‑k with magnitude K and chosen intercept policy).

    print("Fit on raw (with scaling)"); timestamp()
    raw_train = dset_train[[c for c in dset_train.columns if c not in EXCLUDE_COLS]]
    raw_test  = dset_test[[c for c in dset_test.columns if c not in EXCLUDE_COLS]]
    model_raw = build_model(
        raw_train, cv=args.cv, cs=args.cs, use_scaler=True, tol=args.tol,
        solver=args.solver, cv_rule=args.cv_rule, zero_clamp=args.zero_clamp,
        logic_polish=False,  # usually skip logical compression on raw/scaled model
        verbose=True
    )
    print("Done fitting."); timestamp()
    # writing models to disk
    
    print("Saving Fitted models..."); timestamp()

    print("Done saving."); timestamp()


    prob_tr, y_tr, auc_tr, f1_tr = evaluate(model_rect, rt_train)
    prob_te, y_te, auc_te, f1_te = evaluate(model_rect, rt_test)
    print(f"Rectified: train AUC={auc_tr:.3f} | test AUC={auc_te:.3f}")
    print(f"Rectified: train max F1={np.nanmax(f1_tr):.3f} | test max F1={np.nanmax(f1_te):.3f}")

    prob2_tr, y2_tr, auc2_tr, _ = evaluate(model_raw, raw_train)
    prob2_te, y2_te, auc2_te, _ = evaluate(model_raw, raw_test)
    print(f"Raw: train AUC={auc2_tr:.3f} | test AUC={auc2_te:.3f}")
    # ^ In the manuscript’s empirical sections, rectified models are typically
    #   sparser and comparably accurate; sometimes RAW edges out in AUC but at
    #   the cost of interpretability (see UNICEF results discussion).

    # Get PR curve and F1 across thresholds
    prec, rec, thr = precision_recall_curve(y_te.astype(int), prob_te)
    f1_curve = 2 * prec * rec / (prec + rec + 1e-9)
    i_best = int(np.nanargmax(f1_curve))
    t_star = 1 - thr[min(i_best, len(thr) - 1)] # guard for last point
    
    # Confusion at best-F1 threshold
    yhat_star = (prob_te >= t_star)
    tn = int((~y_te & ~yhat_star).sum()); fp = int((~y_te &  yhat_star).sum())
    fn = int(( y_te & ~yhat_star).sum()); tp = int(( y_te &  yhat_star).sum())
    print(f"Best-F1 threshold t*=1 - {t_star:.6e} → TN, FP, FN, TP: {tn} {fp} {fn} {tp} | F1={f1_curve[i_best]:.3f}")
    
    # Confusion at 0.5 (for comparison)
    yhat_05 = (prob_te >= 0.5)
    tn = int((~y_te & ~yhat_05).sum()); fp = int((~y_te &  yhat_05).sum())
    fn = int(( y_te & ~yhat_05).sum()); tp = int(( y_te &  yhat_05).sum())
    print(f"At threshold 0.5 → TN, FP, FN, TP: {tn} {fp} {fn} {tp}")

    highlights = parse_highlights_arg(args.highlights)
    if not highlights:
        highlights = LABELS

    if not args.no_plots:
        f1 = plot_sorted(prob_tr, y_tr, "Case #1 Training Set (Rectified)")
        pd.DataFrame(prob_tr).to_csv("prob_tr.csv", index=False)
        pd.DataFrame(y_tr).to_csv("y_tr.csv", index=False)
        maybe_show_or_save(f1, "Case #1 Training Set (Rectified)", save_dir, args.show_each)

        f2 = plot_sorted(prob_te, y_te, "Case #1 Test Set (Rectified)")
        pd.DataFrame(prob_te).to_csv("prob_te.csv", index=False)
        pd.DataFrame(y_te).to_csv("y_te.csv", index=False)
        maybe_show_or_save(f2, "Case #1 Test Set (Rectified)", save_dir, args.show_each)
        
        f3 = plot_sorted(prob2_tr, y2_tr, "Train Set (Raw)")
        maybe_show_or_save(f3, "Train Set (Raw)", save_dir, args.show_each)

        f4 = plot_sorted(prob2_te, y2_te, "Test Set (Raw)")
        maybe_show_or_save(f4, "Test Set (Raw)", save_dir, args.show_each)

        rect_feat_names = rt_train.drop(columns=[RESP_COL]).columns.tolist()
        coef_rect = coefficient_series_only(model_rect, rect_feat_names)  # no intercept
        f5 = plot_coeff_bars(coef_rect, "Coefficient Magnitudes (Rectified) — labeled true lags", highlights=highlights, annotate=True)
        maybe_show_or_save(f5, "Coefficient Magnitudes (Rectified) — labeled true lags", save_dir, args.show_each)
        # ^ In simulated case (Section 5), this plot reveals precise lag picks for
        #   true variables, contrasting with noisier/dense RAW coefficients.
        
        try:
            lr_rect = model_rect.named_steps["lr"]
            if getattr(lr_rect, "logic_plot", False) and not getattr(lr_rect, "logic_plot_dir", None):
                # No directory was specified → figures live in memory; display/save them now.
                for i, fig in enumerate(getattr(lr_rect, "logic_figs_", []) or [], start=1):
                    title = f"Logical Compression #{i}"
                    maybe_show_or_save(fig, title, save_dir, args.show_each)
        except Exception as _e:
            # Non-fatal: keep going even if there were no figures.
            pass

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
    
        rect_feat_names = rt_train.drop(columns=[RESP_COL]).columns.tolist()
        raw_feat_names  = raw_train.drop(columns=[RESP_COL]).columns.tolist()
    
        save_pipeline_npz(model_rect, rect_feat_names,
                          os.path.join(args.export_dir, "model_rectified.npz"))
        save_limits_json(limits,
                         os.path.join(args.export_dir, "rectify_limits.json"))
        save_pipeline_npz(model_raw,  raw_feat_names,
                          os.path.join(args.export_dir, "model_raw.npz"))
        print(f"[exported] {args.export_dir}")
    
    timestamp("After evaluation")

if __name__ == "__main__":
    main()
