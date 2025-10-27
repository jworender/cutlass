
#!/usr/bin/env python3
"""
Fast script (v4) — fixes for coefficient bar plot:
- Intercept is NOT plotted (only feature coefficients).
- Correct time steps are labeled with the feature name on the bar.
- Non-interactive backend warnings avoided (save instead of show).
- Optional --backend to force a GUI backend (e.g., TkAgg), --save-dir to save PNGs.
- Keeps vectorized rectification and lighter CV for speed.

"""

import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

#–––– Configuration ––––#
RESP_COL     = 'INDC'
EXCLUDE_COLS = ['X', 'UIC', 'iDate', 'iDate.x', 'iDate.y', 'class', 'cols', 'year']

def timestamp(msg=None):
    now = datetime.now().isoformat(timespec="seconds")
    print(f"{msg+': ' if msg else ''}{now}")

def organize(df: pd.DataFrame) -> Dict[str, List[str]]:
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

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _scores_array_from_lrcv(lrcv) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fold-by-C scores and the C grid from a fitted LogisticRegressionCV(refit=False).
    Returns:
      scores_fc : shape (n_folds, n_Cs)     (higher is better since scoring='neg_log_loss')
      Cs        : shape (n_Cs,)
    Works for both ndarray and dict-of-ndarray variants of scikit's scores_.
    """
    Cs = np.asarray(lrcv.Cs_, dtype=float)  # ascending
    scores = lrcv.scores_
    if isinstance(scores, dict):
        # pick one class (binary => one entry)
        scores_fc = next(iter(scores.values()))  # shape (n_folds, n_Cs)
    else:
        # already an array: (n_classes, n_folds, n_Cs) or (n_folds, n_Cs) depending on scikit version
        arr = np.asarray(scores)
        if arr.ndim == 3:
            scores_fc = arr[0, :, :]  # take first class for binary
        else:
            scores_fc = arr           # (n_folds, n_Cs)
    return scores_fc, Cs

def _select_c_by_rule(
    mean_losses: np.ndarray, se_losses: np.ndarray, Cs: np.ndarray,
    rule: Literal["min","1se"] = "min"
) -> Tuple[float, float]:
    """
    Return (C_pick, C_min) given per-C mean loss and SE, using gd_v8's rules:
      - 'min':  pick argmin(mean).
      - '1se':  pick the *most-regularized* C with mean <= min+SE.
    """
    j_min = int(np.argmin(mean_losses))
    if str(rule).lower() == "1se":
        thr = mean_losses[j_min] + se_losses[j_min]
        cand = np.where(mean_losses <= thr)[0]
        j_pick = int(np.min(cand)) if cand.size else j_min
    else:
        j_pick = j_min
    return float(Cs[j_pick]), float(Cs[j_min])

def _select_c_via_1se(
    mean_losses: np.ndarray,
    se_losses: np.ndarray,
    Cs: np.ndarray
) -> Tuple[float, float]:
    """
    Given mean_losses(C), se_losses(C) and the C grid (ascending),
    pick C_1se = smallest C with mean_loss <= min_loss + min_se  (i.e., most-regularized within 1-SE).
    Returns:
      C_1se, C_min
    """
    j_min = int(np.argmin(mean_losses))
    thr = float(mean_losses[j_min] + se_losses[j_min])  # 1-SE threshold
    # candidates whose mean loss within threshold
    cand = np.where(mean_losses <= thr)[0]
    j_1se = int(np.min(cand)) if cand.size else j_min  # most-regularized (smallest C => left-most index)
    return float(Cs[j_1se]), float(Cs[j_min])

def _fit_logregcv_1se_core(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Cs: np.ndarray,
    cv: int = 3,
    tol: float = 1e-3,
    max_iter: int = 2000,
    penalty: str = "l1",
    solver: str = "saga",
    random_state: int = 42,
    scoring: str = "neg_log_loss"
) -> Tuple[LogisticRegression, dict]:
    """
    Core: run LogisticRegressionCV(refit=False), compute 1-SE pick, then refit LogisticRegression at C_1se.
    Returns:
      final_estimator (refit on all data at C_1se),
      info dict with arrays and choices.
    """
    # 1) CV without refit
    lrcv = LogisticRegressionCV(
        Cs=Cs,
        penalty=penalty,
        solver=solver,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        tol=tol,
        max_iter=max_iter,
        refit=False,
        random_state=random_state,
    )
    lrcv.fit(X, y)

    # 2) From scores (higher is better), make losses (lower is better)
    scores_fc, Cs_grid = _scores_array_from_lrcv(lrcv)           # (folds, Cs)
    mean_scores = scores_fc.mean(axis=0)
    # Use ddof=1 if >=2 folds; else zero SE to avoid NaN
    se_scores = (scores_fc.std(axis=0, ddof=1) / np.sqrt(scores_fc.shape[0])) if scores_fc.shape[0] > 1 else np.zeros_like(mean_scores)

    mean_losses = -mean_scores
    se_losses = se_scores  # SE of scores == SE of (-loss) up to sign; threshold uses addition so same values

    C_1se, C_min = _select_c_via_1se(mean_losses, se_losses, Cs_grid)

    # 3) Final refit at C_1se
    final = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=float(C_1se),
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )
    final.fit(X, y)

    info = dict(
        Cs=Cs_grid,
        mean_losses=mean_losses,
        se_losses=se_losses,
        C_min=C_min,
        C_1se=C_1se,
    )
    return final, info

def _gd_like_kfold_indices(n: int, cv: int, random_state: int) -> list[np.ndarray]:
    """
    Mirror gd_v8: rng.shuffle(np.arange(n)) then np.array_split into cv folds.
    """
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, cv)  # list of fold-index arrays

def _binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())

def rectify_fast(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    sdfilter: Optional[float] = 3.0,
    snap: float = 0.001,
):
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

def _is_binary_pm1(dfX: pd.DataFrame) -> bool:
    cols = dfX.columns[:min(len(dfX.columns), 10)]
    vals = np.unique(dfX[cols].to_numpy().ravel())
    return set(vals.tolist()).issubset({-1, 1})

def _cv_losses_sklearn_no_warm(
    X_df: pd.DataFrame,
    *,
    y_col: str = "INDC",
    use_scaler: bool,
    Cs: np.ndarray,
    cv: int,
    random_state: int,
    solver: str = "saga",
    tol: float = 1e-3,
    max_iter: int = 2000
) -> Tuple[np.ndarray, list[np.ndarray]]:
    """
    For each fold and each C, fit a *fresh* LogisticRegression (no warm start),
    then compute validation neg. log-loss. Returns (fold_losses, folds),
    where fold_losses has shape (cv, nC). This emulates your GD CV routine’s
    loss table, without any sklearn path warm-start advantages.
    """
    # Extract arrays
    X = X_df.drop(columns=[y_col]).to_numpy(dtype=np.float64, copy=False)
    y = X_df[y_col].astype(int).to_numpy()
    n = X.shape[0]
    folds = _gd_like_kfold_indices(n, cv, random_state)

    fold_losses = np.empty((cv, len(Cs)), dtype=np.float64)

    for f, val_idx in enumerate(folds):
        mask = np.ones(n, dtype=bool); mask[val_idx] = False
        Xtr, ytr = X[mask, :], y[mask]
        Xva, yva = X[val_idx, :], y[val_idx]

        # Optional scaling for raw view only
        if use_scaler:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xva_s = scaler.transform(Xva)
        else:
            Xtr_s, Xva_s = Xtr, Xva

        # For each C, fit *from scratch* and score on validation
        for j, Cj in enumerate(Cs):
            lr = LogisticRegression(
                penalty="l1", solver=solver, C=float(Cj),
                tol=tol, max_iter=max_iter, random_state=random_state,
                warm_start=False, fit_intercept=True
            )
            lr.fit(Xtr_s, ytr)
            pva = lr.predict_proba(Xva_s)[:, 1]
            fold_losses[f, j] = _binary_log_loss(yva.astype(float), pva)

    return fold_losses, folds

def sklearn_build_model_via_external_cv(
    X_df: pd.DataFrame,
    *,
    y_col: str = "INDC",
    use_scaler: bool,
    cv: int = 3,
    cs: int = 15,
    c_lo: float = -3.0,
    c_hi: float =  3.0,
    random_state: int = 42,
    solver: str = "saga",
    tol: float = 1e-3,
    max_iter: int = 2000,
    cv_rule: Literal["min","1se"] = "min"
) -> Tuple[Pipeline, dict]:
    """
    External CV (no warm start) over logspace(C), apply gd_v8's 'min' or '1se' rule,
    then refit a final LogisticRegression at the chosen C on *all* data.
    Returns (sklearn Pipeline, info dict).
    """
    Cs = np.logspace(c_lo, c_hi, cs).astype(float)

    # 1) Get per-fold, per-C validation losses (no warm-start)
    fold_losses, folds = _cv_losses_sklearn_no_warm(
        X_df, y_col=y_col, use_scaler=use_scaler,
        Cs=Cs, cv=cv, random_state=random_state,
        solver=solver, tol=tol, max_iter=max_iter
    )

    # 2) Aggregate and pick C by the same rule as gd_v8
    mean_losses = fold_losses.mean(axis=0)
    se_losses   = (fold_losses.std(axis=0, ddof=1) / np.sqrt(cv)) if cv > 1 else np.zeros_like(mean_losses)
    C_pick, C_min = _select_c_by_rule(mean_losses, se_losses, Cs, rule=cv_rule)

    # 3) Final refit on *all* data at C_pick (no scaler for rectified)
    X = X_df.drop(columns=[y_col]).to_numpy(dtype=np.float64, copy=False)
    y = X_df[y_col].astype(int).to_numpy()

    steps = []
    if use_scaler:
        scaler = StandardScaler(with_mean=True, with_std=True)
        steps.append(("scaler", scaler))
        X_s = scaler.fit_transform(X)
    else:
        X_s = X

    final_lr = LogisticRegression(
        penalty="l1", solver=solver, C=float(C_pick),
        tol=tol, max_iter=max_iter, random_state=random_state,
        warm_start=False, fit_intercept=True
    )
    final_lr.fit(X_s, y)
    steps.append(("lr", final_lr))
    pipe = Pipeline(steps)

    info = dict(
        Cs=Cs, cv=cv, cv_rule=cv_rule,
        mean_losses=mean_losses, se_losses=se_losses,
        C_min=C_min, C_pick=C_pick,
        folds=[fold.tolist() for fold in folds]
    )
    return pipe, info

def sklearn_build_model_1se(
    X_df: pd.DataFrame,
    *,
    y_col: str = "INDC",
    use_scaler: bool = False,       # False for rectified ±1; True for raw
    cv: int = 3,
    cs: int = 15,
    c_lo: float = -3.0,             # use (-4, +4) if you want parity with gd_v8
    c_hi: float =  3.0,
    tol: float = 1e-3,
    max_iter: int = 2000,
    solver: str = "saga",
    random_state: int = 42
) -> Tuple[Pipeline, dict]:
    """
    Build a sklearn Pipeline (optional StandardScaler -> LogisticRegression at C_1se)
    using the 1-SE rule over a logspace grid. Returns (pipeline, info).
    The 'info' dict contains C_min, C_1se, Cs, mean_losses, se_losses.
    """
    Xonly = X_df.drop(columns=[y_col]).to_numpy(dtype=np.float64, copy=False)
    y     = X_df[y_col].astype(int).to_numpy()

    Cs = np.logspace(c_lo, c_hi, cs).astype(float)

    # Fit CV, compute 1-SE, refit LogisticRegression at C_1se
    final_est, info = _fit_logregcv_1se_core(
        Xonly, y,
        Cs=Cs, cv=cv, tol=tol, max_iter=max_iter,
        penalty="l1", solver=solver, random_state=random_state,
        scoring="neg_log_loss"
    )

    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("lr", final_est))
    pipe = Pipeline(steps)
    # NOTE: pipeline.fit here is unnecessary (final_est is already fit) but harmless if you prefer:
    # pipe.fit(Xonly, y)
    return pipe, info

def sklearn_fit_1se_rectified(rt_train: pd.DataFrame, **kwargs) -> Tuple[Pipeline, dict]:
    """
    1-SE model on rectified ±1 features (no scaler).
    kwargs forwarded to sklearn_build_model_1se (cv, cs, c_lo/c_hi, tol, max_iter, solver, random_state).
    """
    return sklearn_build_model_1se(rt_train, use_scaler=False, **kwargs)

def sklearn_fit_1se_raw(raw_train: pd.DataFrame, **kwargs) -> Tuple[Pipeline, dict]:
    """
    1-SE model on raw continuous features (with StandardScaler).
    kwargs forwarded to sklearn_build_model_1se (cv, cs, c_lo/c_hi, tol, max_iter, solver, random_state).
    """
    return sklearn_build_model_1se(raw_train, use_scaler=True, **kwargs)

def sklearn_build_rectified_external_cv(rt_train: pd.DataFrame, **kw):
    # Rectified ±1 view → no scaling
    return sklearn_build_model_via_external_cv(rt_train, use_scaler=False, **kw)

def sklearn_build_raw_external_cv(raw_train: pd.DataFrame, **kw):
    # Raw continuous view → scale
    return sklearn_build_model_via_external_cv(raw_train, use_scaler=True, **kw)

def build_model(
    X: pd.DataFrame,
    y_col: str = RESP_COL,
    cv: int = 3,
    cs: int = 15,
    use_scaler: Optional[bool] = None,
    solver: str = "saga",
    tol: float = 1e-3,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    Xonly = X.drop(columns=[y_col])
    y = X[y_col].astype(int).to_numpy()
    is_bin = _is_binary_pm1(Xonly) if use_scaler is None else False
    do_scale = (use_scaler if use_scaler is not None else not is_bin)

    Cs = np.logspace(-4, 4, cs)
    lr_cv = LogisticRegressionCV(
        Cs=Cs,
        penalty="l1",
        solver=solver,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        tol=tol,
        max_iter=max_iter,
        refit=True,
        random_state=random_state,
    )
    steps = []
    if do_scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("lr", lr_cv))
    pipe = Pipeline(steps)
    pipe.fit(Xonly.to_numpy(dtype=np.float64, copy=False), y)
    return pipe

def evaluate(pipe: Pipeline, X: pd.DataFrame, y_col: str = RESP_COL):
    Xonly = X.drop(columns=[y_col])
    y = X[y_col].astype(int).to_numpy()
    prob = pipe.predict_proba(Xonly.to_numpy(dtype=np.float64, copy=False))[:, 1]
    auc = roc_auc_score(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    fscore = 2 * rec * prec / (rec + prec + 1e-9)
    return prob, y.astype(bool), auc, fscore

def plot_sorted(prob: np.ndarray, y_bool: np.ndarray, title: str, h: float = 0.5):
    import matplotlib.pyplot as plt
    order = np.argsort(prob)
    fig = plt.figure(num=title)
    ax = fig.gca()
    sc = ax.scatter(np.arange(prob.size), prob[order], c=y_bool[order], s=6, alpha=0.6, rasterized=True)
    ax.axhline(h, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sorted Examples")
    ax.set_ylabel("Model Response")
    fig.tight_layout()
    return fig

def coefficient_series_only(pipe: Pipeline, feature_names: List[str]):
    """
    Return only the coefficients (no intercept) as a Series aligned to feature_names.
    """
    lr = pipe.named_steps["lr"]
    coef = lr.coef_.ravel()
    if coef.shape[0] != len(feature_names):
        raise ValueError(f"coef length {coef.shape[0]} != feature_names length {len(feature_names)}")
    return pd.Series(coef, index=feature_names)

def plot_coeff_bars(beta: pd.Series, title: str, highlights: Optional[List[str]] = None,
                    annotate: bool = True, fontsize: int = 9):
    """
    Bar plot of coefficients (no intercept). If `highlights` is provided, annotate those
    coefficients with the feature name at their bar location.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(num=title)
    ax = fig.gca()
    x = np.arange(len(beta))
    bars = ax.bar(x, beta.values, linewidth=1)

    # nice symmetric y-limits with a little headroom for labels
    m = np.nanmax(np.abs(beta.values)) if len(beta) else 1.0
    ylim = max(m * 1.3, 0.5)
    ax.set_ylim(-0.1*ylim, ylim)

    # Annotate highlighted feature names
    if highlights:
        # For each pattern, find feature(s) that endwith the pattern and label
        for pat in highlights:
            matches = [i for i, nm in enumerate(beta.index) if nm.endswith(pat)]
            for i in matches:
                name = beta.index[i]
                val  = beta.values[i]
                # place label slightly above or below the bar depending on sign
                offset = 0.04 * ylim
                y = val + (offset if val >= 0 else -offset)
                va = 'bottom' if val >= 0 else 'top'
                ax.text(i, y, name, rotation=90, ha='center', va=va, fontsize=fontsize)

    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig

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
            # fallback: non-interactive backend, auto-save
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".", help="Directory containing case_1_train_data.csv and case_1_test_data.csv")
    ap.add_argument("--cv", type=int, default=3, help="n-fold cross validation")
    ap.add_argument("--cs", type=int, default=25, help="number of C values to try (log-spaced)")
    ap.add_argument("--sdfilter", type=float, default=3.0, help="sdfilter used in rectification")
    ap.add_argument("--snap", type=float, default=0.001, help="snap threshold used in rectification")
    ap.add_argument("--tol", type=float, default=1e-3, help="solver tolerance")
    ap.add_argument("--save-dir", type=str, default="", help="If set, save all figures to this directory")
    ap.add_argument("--show-each", action="store_true", help="Call plt.show() after EACH figure (prevents only-last-figure issue)")
    ap.add_argument("--no-plots", action="store_true", help="skip plotting")
    ap.add_argument("--backend", type=str, default="", help="Matplotlib backend to use (e.g., TkAgg, QtAgg). If empty, use default.")
    ap.add_argument("--highlights", type=str, default="", help="Comma-separated list of feature suffixes to highlight & label (e.g., 'V5TM10,V6TM6,...)'")
    args = ap.parse_args()

    # Backend selection BEFORE importing pyplot anywhere else
    if args.backend:
        try:
            matplotlib.use(args.backend, force=True)
            print(f"[matplotlib] Using backend: {args.backend}")
        except Exception as e:
            print(f"[matplotlib] Could not set backend to {args.backend}: {e}")

    # Ensure save dir exists
    save_dir = args.save_dir if args.save_dir else None
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    timestamp("Start")

    # Load
    tr_path = os.path.join(args.data_dir, "case_1_train_data.csv")
    te_path = os.path.join(args.data_dir, "case_1_test_data.csv")
    dset_train = pd.read_csv(tr_path, true_values=["TRUE", "True", "T", "1"], false_values=["FALSE", "False", "F", "0"])
    dset_test  = pd.read_csv(te_path, true_values=["TRUE", "True", "T", "1"], false_values=["FALSE", "False", "F", "0"])

    # Organize groups
    print("Detecting groups...")
    groups = organize(dset_train)
    print("Done.")

    # Rectify once (train) and apply same limits (test)
    print("Rectify (vectorized) start"); timestamp()
    rt_train, limits = rectify_fast(dset_train, groups, limits=None, sdfilter=args.sdfilter, snap=args.snap)
    rt_test,  _      = rectify_fast(dset_test,  groups, limits=limits,    sdfilter=args.sdfilter, snap=args.snap)
    print("Rectify stop"); timestamp()
    raw_train = dset_train[[c for c in dset_train.columns if c not in EXCLUDE_COLS]]
    raw_test  = dset_test[[c for c in dset_test.columns if c not in EXCLUDE_COLS]]

    # Build models (rectified & raw)
    #print("Fit on rectified (skip scaling)"); timestamp()
    #model_rect = build_model(rt_train, cv=args.cv, cs=args.cs, use_scaler=False, tol=args.tol)
    #print("Fit on raw (with scaling)"); timestamp()
    #model_raw = build_model(raw_train, cv=args.cv, cs=args.cs, use_scaler=True, tol=args.tol)
    
    # Evaluate / plot
    # Rectified (sklearn, 1-SE)
    sk_rect_pipe, sk_rect_info = sklearn_fit_1se_rectified(
        rt_train,
        cv=args.cv, cs=args.cs, c_lo=-4, c_hi=4,  # use (-4,4) to match gd_v8 grid if you want exact parity
        tol=1e-3, max_iter=10000, solver="saga", random_state=42
    )
    # Raw (sklearn, 1-SE)
    sk_raw_pipe, sk_raw_info = sklearn_fit_1se_raw(
        raw_train,
        cv=args.cv, cs=args.cs, c_lo=-4, c_hi=4,
        tol=1e-3, max_iter=10000, solver="saga", random_state=42
    )
    print("Done fitting"); timestamp()

    prob_tr, y_tr, auc_tr, f1_tr = evaluate(sk_rect_pipe, rt_train)  # reuse your existing evaluate(...)
    prob_te, y_te, auc_te, f1_te = evaluate(sk_rect_pipe, rt_test)  # reuse your existing evaluate(...)

    #prob_tr, y_tr, auc_tr, f1_tr = evaluate(model_rect, rt_train)
    #prob_te, y_te, auc_te, f1_te = evaluate(model_rect, rt_test)
    print(f"Rectified: train AUC={auc_tr:.3f} | test AUC={auc_te:.3f}")
    print(f"Rectified: train max F1={np.nanmax(f1_tr):.3f} | test max F1={np.nanmax(f1_te):.3f}")

    prob2_tr, y2_tr, auc2_tr, _ = evaluate(sk_raw_pipe, raw_train)
    prob2_te, y2_te, auc2_te, _ = evaluate(sk_raw_pipe, raw_test)

    #prob2_tr, y2_tr, auc2_tr, _ = evaluate(model_raw, raw_train)
    #prob2_te, y2_te, auc2_te, _ = evaluate(model_raw, raw_test)
    print(f"Raw: train AUC={auc2_tr:.3f} | test AUC={auc2_te:.3f}")

    highlights = parse_highlights_arg(args.highlights)
    # Default list for Case 1 if none provided
    if not highlights:
        highlights = ["V5TM10","V6TM6","V9TM1","V10TM0","V15TM5","V25TM3","V30TM7"]

    # Plotting
    if not args.no_plots:
        f1 = plot_sorted(prob_tr, y_tr, "Case #1 Training Set (Rectified)")
        maybe_show_or_save(f1, "Case #1 Training Set (Rectified)", save_dir, args.show_each)

        f2 = plot_sorted(prob_te, y_te, "Case #1 Test Set (Rectified)")
        maybe_show_or_save(f2, "Case #1 Test Set (Rectified)", save_dir, args.show_each)

        f3 = plot_sorted(prob2_tr, y2_tr, "Train Set (Raw)")
        maybe_show_or_save(f3, "Train Set (Raw)", save_dir, args.show_each)

        f4 = plot_sorted(prob2_te, y2_te, "Test Set (Raw)")
        maybe_show_or_save(f4, "Test Set (Raw)", save_dir, args.show_each)

        # Coefficient bar plot (RECTIFIED model)
        rect_feat_names = rt_train.drop(columns=[RESP_COL]).columns.tolist()
        coef_rect = coefficient_series_only(sk_rect_pipe, rect_feat_names)
        f5 = plot_coeff_bars(
            coef_rect,
            "Coefficient Magnitudes (Rectified) — labeled true lags",
            highlights=highlights,
            annotate=True
        )
        maybe_show_or_save(f5, "Coefficient Magnitudes (Rectified) — labeled true lags", save_dir, args.show_each)

    timestamp("After evaluation")

if __name__ == "__main__":
    main()
