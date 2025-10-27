"""
Low-level solvers for L1-regularised logistic regression.

The implementations are adapted from the research prototypes used in the
experiments and retain the warm-start and {-1, +1} specific optimisations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._math import (
    _binary_log_loss_from_logits,
    _sigmoid,
    _soft_threshold,
)

__all__ = ["_CDLogistic", "_FISTALogistic"]


class _CDLogistic:
    """
    Coordinate Descent with Sequential Strong Rules and KKT screening.
    """

    def __init__(
        self,
        lam: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 2000,
        verbose: bool = False,
        kkt_tol: float = 1e-4,
        all_pm1: bool = False,
    ) -> None:
        self.lam = float(lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.kkt_tol = float(kkt_tol)
        self.all_pm1 = bool(all_pm1)
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.n_iter_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        w0: Optional[np.ndarray] = None,
        b0: Optional[float] = None,
        lam_prev: Optional[float] = None,
        active_init: Optional[np.ndarray] = None,
    ) -> "_CDLogistic":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n, p = X.shape
        w = np.zeros(p, dtype=np.float64) if w0 is None else w0.astype(np.float64).copy()

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

        if lam_prev is None or lam_prev <= 0:
            strong_thr = lam
        else:
            strong_thr = max(0.0, 2 * lam - lam_prev)

        grad = (X.T @ (p_hat - y)) / max(float(n), 1.0)

        if active_init is None:
            active_set = set(np.where(np.abs(grad) >= strong_thr)[0].tolist())
        else:
            active_set = set(int(i) for i in np.asarray(active_init).tolist())

        full_cycle = np.arange(p, dtype=np.int32)
        diag_h = np.mean(wght) if all_pm1 else np.mean((X ** 2) * wght[:, None], axis=0)
        diag_h[diag_h <= 1e-12] = 1e-12

        for it in range(1, self.max_iter + 1):
            order = sorted(active_set) if active_set else []
            if len(order) != p:
                order.extend(int(i) for i in full_cycle if i not in active_set)

            max_dw = 0.0
            for j in order:
                r_j = ((p_hat - y) @ X[:, j]) / max(float(n), 1.0)
                if w[j] != 0.0:
                    grad_j = r_j + lam * np.sign(w[j])
                else:
                    grad_j = np.clip(r_j, -lam, lam)

                z_old = w[j]
                w_j = _soft_threshold(z_old - r_j / diag_h[j], lam / diag_h[j])
                dw = w_j - z_old
                if dw != 0.0:
                    z += dw * X[:, j]
                    p_hat = _sigmoid(z)
                    wght = p_hat * (1 - p_hat)
                    if not all_pm1:
                        diag_h[j] = np.mean((X[:, j] ** 2) * wght)
                        diag_h[j] = max(diag_h[j], 1e-12)
                w[j] = w_j
                max_dw = max(max_dw, abs(dw))

            grad_b = np.mean(p_hat - y)
            b -= grad_b
            z -= grad_b

            grad = (X.T @ (p_hat - y)) / max(float(n), 1.0)
            inactive = [j for j in range(p) if j not in active_set]
            viol = np.where(np.abs(grad[inactive]) >= lam + self.kkt_tol)[0]
            for idx in viol:
                active_set.add(inactive[idx])

            if max_dw <= self.tol and np.abs(grad_b) <= self.tol:
                self.n_iter_ = it
                break
            self.n_iter_ = it

        self.w_ = w
        self.b_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        z = X @ self.w_ + self.b_
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])


class _FISTALogistic:
    """Accelerated proximal gradient (FISTA) for logistic regression."""

    def __init__(
        self,
        lam: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 4000,
        step: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.lam = float(lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.step = step
        self.verbose = bool(verbose)
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.n_iter_: int = 0

    @staticmethod
    def _estimate_L(X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        v = np.random.default_rng(123).standard_normal(p)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(12):
            Xv = X @ v
            v = X.T @ Xv
            nv = np.linalg.norm(v)
            if nv == 0:
                break
            v /= nv
        smax_sq = np.linalg.norm(X @ v) ** 2
        return 0.25 * smax_sq / max(n, 1) + 0.25

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        w0: Optional[np.ndarray] = None,
        b0: Optional[float] = None,
    ) -> "_FISTALogistic":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n, p = X.shape
        w = np.zeros(p, dtype=np.float64) if w0 is None else w0.astype(np.float64).copy()

        if b0 is None:
            py = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
            b = float(np.log(py / (1 - py)))
        else:
            b = float(b0)

        L = self._estimate_L(X) if self.step is None else 1.0 / self.step
        tstep = 0.9 / L

        w_y = w.copy()
        b_y = b
        t = 1.0
        prev_obj = np.inf

        for it in range(1, self.max_iter + 1):
            z = X @ w_y + b_y
            p_hat = _sigmoid(z)
            grad_w = (X.T @ (p_hat - y)) / max(float(n), 1.0)
            grad_b = float(np.sum(p_hat - y)) / max(float(n), 1.0)

            found = False
            bt = 0
            while not found and bt < 20:
                w_new = _soft_threshold(w_y - tstep * grad_w, tstep * self.lam)
                b_new = b_y - tstep * grad_b

                z_new = X @ w_new + b_new
                obj_new = _binary_log_loss_from_logits(y, z_new) + self.lam * np.sum(np.abs(w_new))

                dz_w = w_new - w_y
                dz_b = b_new - b_y
                quad = (
                    _binary_log_loss_from_logits(y, z)
                    + np.dot(grad_w, dz_w)
                    + grad_b * dz_b
                    + (np.linalg.norm(dz_w) ** 2 + dz_b * dz_b) / (2 * tstep)
                )
                if obj_new <= quad + 1e-12:
                    found = True
                else:
                    tstep *= 0.5
                    bt += 1

            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            w_acc = w_new + ((t - 1) / t_new) * (w_new - w)
            b_acc = b_new + ((t - 1) / t_new) * (b_new - b)
            if np.dot((w_acc - w_new), (w_new - w)) + (b_acc - b_new) * (b_new - b) > 0:
                w_y, b_y = w_new, b_new
                t = 1.0
            else:
                w_y, b_y = w_acc, b_acc
                t = t_new

            dw = np.linalg.norm(w_new - w)
            db = abs(b_new - b)
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        z = X @ self.w_ + self.b_
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

