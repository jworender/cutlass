#!/usr/bin/env python3

"""
Automates experiments for the paper:
  "Anytime, Performance-Bounded Rule Compression via Youden-Target Logical Polishing"

This variant consumes existing CSV datasets instead of generating synthetic data.
It reuses helper modules for:
  - organize(...), rectify_fast(...), build_model(...), evaluate(...), calculate_youden_j(...)

Outputs per configuration:
  * CSV rows in results/summary.csv with metrics and chosen rule size k
  * JSON diagnostics per run (including J-vs-k arrays, chosen k, adoption decision)
  * Model and rectification artifacts (optional)
  * Plots produced by the logic-polish routine (saved under the run directory)

Usage (quick start):
  python experiment_driver_csv.py \
      --outdir runs/exp1 \
      --seeds 1234 2027 \
      --dataset-csv sample_data/leukemia_gene_expression.csv \
      --tr-fracs 0.50 0.30 \
      --targets 0.90 0.95 0.97 \
      --k-policies global premin \
      --intercepts mean mofk maxj \
      --Ks 5 10 20
"""
from __future__ import annotations
if __name__ == "__main__":
    print("Loading production libraries...")
import argparse, json, time, itertools, importlib.util, math, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from cutlass import (
    CutlassLogisticCV,
    Rectifier,
    StandardScaler,
    calculate_youden_j,
    precision_recall_curve,
    roc_auc_score,
)
from cutlass.pipeline import Pipeline

# ---------- simple function to print timestamp to console -----------
def timestamp():
    strtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"({strtime})")

# ---------- robust import by file path (no renames needed) ----------
HERE = Path(__file__).resolve().parent
SK_PATH = HERE / "case_1_simple_script_scikit_fast_v6.py"

def _import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    # --- critical line: make the module discoverable for pickling/unpickling ---
    sys.modules[name] = mod
    # ---------------------------------------------------------------------------
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
  
if __name__ == "__main__":
    print("Loading custom libraries ", end=""); timestamp()
skcase = _import_from_path("case_1_simple_script_scikit_fast_v6", SK_PATH)
# convenience aliases to your functions/classes
sklearn_build_rectified_external_cv = skcase.sklearn_build_rectified_external_cv
sklearn_build_raw_external_cv = skcase.sklearn_build_raw_external_cv

RESP = "INDC"
EXCL = {"X", "UIC", "iDate", "iDate.x", "iDate.y", "class", "cols", "year"}
LABELS = ["V5TM10", "V6TM6", "V9TM1", "V10TM0", "V15TM5", "V25TM3", "V30TM7"]
youdenJ = calculate_youden_j
EXCLUDE_FOR_RECT = sorted(set(EXCL) | {RESP})

# ---------- dataset helpers ----------

def _sanitize_tag(tag: str) -> str:
    """
    Make a filesystem-friendly identifier for use in run directories.
    """
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag or "")
    clean = clean.strip("_")
    return clean or "dataset"


def _prepare_train_test(
    *,
    seed: int,
    dataset_csv: Optional[Path],
    train_csv: Optional[Path],
    test_csv: Optional[Path],
    train_frac: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, float, int, bool]:
    """
    Load train/test dataframes based on user-supplied CSVs.

    Returns (train_df, test_df, dataset_tag, actual_train_frac, total_examples, used_explicit_split)
    """
    if dataset_csv is not None and (train_csv is not None or test_csv is not None):
        raise ValueError("Provide either dataset_csv or (train_csv and test_csv), not both.")

    explicit_split = train_csv is not None or test_csv is not None

    if dataset_csv is not None:
        dataset_csv = Path(dataset_csv)
        if train_frac is None:
            raise ValueError("train_frac must be provided when using a single dataset CSV.")
        if not 0 < train_frac < 1:
            raise ValueError(f"train_frac must be between 0 and 1 (exclusive); got {train_frac}.")

        df_full = pd.read_csv(dataset_csv)
        n_total = len(df_full)
        if n_total < 2:
            raise ValueError(f"Dataset {dataset_csv} must contain at least 2 rows.")

        n_train = int(round(train_frac * n_total))
        if n_train <= 0 or n_train >= n_total:
            raise ValueError(
                f"train_frac={train_frac} leads to invalid split (train size {n_train}, total {n_total})."
            )

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        train_df = df_full.iloc[train_idx].reset_index(drop=True).copy()
        test_df = df_full.iloc[test_idx].reset_index(drop=True).copy()
        actual_frac = len(train_df) / n_total
        dataset_tag = _sanitize_tag(dataset_csv.stem)
        return train_df, test_df, dataset_tag, actual_frac, n_total, False

    if train_csv is None or test_csv is None:
        raise ValueError("Must supply either dataset_csv or both train_csv and test_csv.")

    train_csv = Path(train_csv)
    test_csv = Path(test_csv)

    train_df = pd.read_csv(train_csv).reset_index(drop=True).copy()
    test_df = pd.read_csv(test_csv).reset_index(drop=True).copy()

    if train_frac is not None:
        print(
            f"[warning] Training fraction {train_frac} ignored because explicit train/test CSVs were provided.",
            flush=True,
        )

    n_total = len(train_df) + len(test_df)
    if n_total == 0:
        raise ValueError("Combined train/test CSVs are empty.")
    if len(train_df) == 0 or len(test_df) == 0:
        print("[warning] One of the CSV splits is empty; downstream models may fail.", flush=True)

    actual_frac = len(train_df) / n_total

    # Prefer the parent directory name (e.g., run folder) if available.
    parent = train_csv.resolve().parent
    dataset_tag_candidate = parent.parent.name if parent.name == "raw_splits" else parent.name
    if not dataset_tag_candidate:
        dataset_tag_candidate = train_csv.stem
    dataset_tag = _sanitize_tag(dataset_tag_candidate)

    return train_df, test_df, dataset_tag, actual_frac, n_total, True

# ---------- plotting + reporting helpers ----------

def plot_sorted(prob, y_bool, title, h=0.5):
    # Primary key: probability (ascending). Tie-break on label so negatives come first.
    order = np.lexsort((y_bool.astype(int), prob))
    fig, ax = plt.subplots()
    ax.scatter(np.arange(prob.size), prob[order], c=y_bool[order], s=6, alpha=0.8, rasterized=True)
    ax.axhline(h, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sorted Examples (tie-broken)")
    ax.set_ylabel("Model Response")
    fig.tight_layout()
    return fig


def plot_sorted_tiebreak(prob, y_bool, title, h=0.5):
    # Sort by probability; for exact ties put negatives first.
    order = np.lexsort((~y_bool, prob))
    fig, ax = plt.subplots()
    ax.scatter(np.arange(prob.size), prob[order], c=y_bool[order], s=6, alpha=0.8, rasterized=True)
    ax.axhline(h, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sorted Examples (tie-broken: negatives first)")
    ax.set_ylabel("Model Response")
    fig.tight_layout()
    return fig


def maybe_show_or_save(fig, title: str, save_dir: Optional[str], show_each: bool):
    backend = matplotlib.get_backend().lower()
    interactive = any(key in backend for key in ("qt", "tk", "wx", "gtk", "macosx", "nbagg", "ipympl"))
    if save_dir:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_")
        out = os.path.join(save_dir, f"{safe}.png")
        fig.savefig(out, dpi=150)
        print(f"[saved] {out}")
        plt.close(fig)
        return

    if interactive:
        if show_each:
            try:
                fig.canvas.manager.set_window_title(title)
            except Exception:
                pass
            plt.show()
        else:
            fig.show()
        return

    auto_dir = os.path.abspath("./figs")
    os.makedirs(auto_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_")
    out = os.path.join(auto_dir, f"{safe}.png")
    fig.savefig(out, dpi=150)
    print(f"[non-interactive backend: saved] {out}")
    plt.close(fig)


def _natural_vtm_key(name: str):
    """
    Parse names like 'V5TM10' -> (5, 10, 'V5TM10') for a natural sort by variable then lag.
    Unparsable names go to the end.
    """
    match = re.match(r"^V(\d+)TM(\d+)$", name)
    if match:
        return (int(match.group(1)), int(match.group(2)), name)
    return (10**9, 10**9, name)


def coefficient_series_only(pipe: Pipeline, feature_names: List[str], *, sort_for_plot: bool = False) -> pd.Series:
    """
    Return coefficients as a Series indexed by feature_names.
    If sort_for_plot=True, return a reordered copy that is easier to read for plotting.
    """
    lr = pipe.named_steps["lr"]
    coef = lr.coef_.ravel()
    if coef.shape[0] != len(feature_names):
        raise ValueError(f"coef length {coef.shape[0]} != feature_names length {len(feature_names)}")
    series = pd.Series(coef, index=feature_names)
    if sort_for_plot:
        series = series.sort_index(key=lambda idx: [_natural_vtm_key(name) for name in idx])
    return series


def plot_coeff_bars(
    beta: pd.Series,
    title: str,
    highlights: Optional[List[str]] = None,
    *,
    annotate: bool = True,
    fontsize: int = 9,
    exact_match: bool = True,
):
    """Bar plot of coefficients with optional highlight labels."""
    fig = plt.figure(num=title)
    ax = fig.gca()

    x = np.arange(len(beta))
    ax.bar(x, beta.values, linewidth=1)

    max_abs = np.nanmax(np.abs(beta.values)) if len(beta) else 1.0
    ylim = max(max_abs * 1.3, 0.5)
    ax.set_ylim(-0.1 * ylim, ylim)

    if annotate and highlights:
        if exact_match:
            patterns = [re.compile(rf"^{re.escape(pat)}$") for pat in highlights]

            def is_hit(name: str) -> bool:
                return any(rx.match(name) for rx in patterns)
        else:

            def is_hit(name: str) -> bool:
                return any(pat in name for pat in highlights)

        for idx, name in enumerate(beta.index):
            if is_hit(name):
                val = beta.values[idx]
                offset = 0.04 * ylim
                y_coord = val + (offset if val >= 0 else -offset)
                valign = "bottom" if val >= 0 else "top"
                ax.text(idx, y_coord, name, rotation=90, ha="center", va=valign, fontsize=fontsize)

    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig

# ---------- helpers ----------

def organize(df: pd.DataFrame) -> Dict[str, List[str]]:
    feats = [str(c) for c in df.columns if c not in EXCL and c != RESP]
    groups: Dict[str, List[str]] = {}
    for feat in feats:
        prefix = re.sub(r"\d+$", "", feat)
        groups.setdefault(prefix, []).append(feat)
    for prefix, cols in groups.items():
        numeric: List[Tuple[int, str]] = []
        non_numeric: List[str] = []
        for col in cols:
            match = re.search(r"(\d+)$", col)
            if match:
                numeric.append((int(match.group(1)), col))
            else:
                non_numeric.append(col)
        ordered = [name for _, name in sorted(numeric)]
        ordered.extend(non_numeric)
        groups[prefix] = ordered
    return groups


def _rectify_with_cutlass(
    df: pd.DataFrame,
    *,
    rectifier: Rectifier,
) -> pd.DataFrame:
    X_rect = rectifier.transform(df)
    rect_df = pd.DataFrame(X_rect, columns=rectifier.feature_names_, index=df.index)
    rect_df[RESP] = df[RESP].values
    return rect_df


def rectify_fast(
    df: pd.DataFrame,
    groups: Dict[str, Sequence[str]],
    *,
    rectifier: Optional[Rectifier] = None,
    sdfilter: Optional[float] = 3.0,
    snap: float = 0.001,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Tuple[float, float]]], Rectifier]:
    if rectifier is None:
        rectifier = Rectifier(
            groups=groups,
            sdfilter=sdfilter,
            snap=snap,
            exclude_features=EXCLUDE_FOR_RECT,
        )
        X_rect = rectifier.fit_transform(
            df, df[RESP].astype(int).to_numpy(copy=False)
        )
        rect_df = pd.DataFrame(
            X_rect, columns=rectifier.feature_names_, index=df.index
        )
        rect_df[RESP] = df[RESP].values
    else:
        rect_df = _rectify_with_cutlass(df, rectifier=rectifier)
    return rect_df, rectifier.limits_, rectifier


def _is_binary_pm1(dfX: pd.DataFrame) -> bool:
    if dfX.empty:
        return True
    sample_cols = dfX.columns[: min(len(dfX.columns), 10)]
    vals = np.unique(dfX[sample_cols].to_numpy(dtype=np.float64, copy=False))
    return np.all(np.isin(vals, (-1, 1)))


def build_model(
    X: pd.DataFrame,
    y_col: str = RESP,
    cv: int = 3,
    cs: int | Sequence[float] = 15,
    use_scaler: Optional[bool] = None,
    solver: str = "hybrid",
    tol: float = 1e-4,
    max_iter: int = 2000,
    random_state: int = 42,
    cv_rule: str = "min",
    zero_clamp: float = 0.0,
    logic_polish: bool = True,
    logic_scale: float = 10.0,
    logic_target: Optional[float] = None,
    logic_maxk: Optional[int] = None,
    logic_rel_tol: float = 0.2,
    logic_plot: bool = True,
    logic_plot_dir: Optional[str] = None,
    logic_Ks_plot: Optional[Sequence[float]] = None,
    logic_k_policy: str = "global",
    logic_smooth_k: int = 7,
    logic_firstmin_drop: float = 0.05,
    logic_firstmin_frac: float = 0.5,
    logic_intercept: str = "mofk",
    logic_m: Optional[int] = None,
    logic_m_frac: Optional[float] = None,
    verbose: bool = True,
) -> Pipeline:
    Xonly = X.drop(columns=[y_col])
    y = X[y_col].astype(int).to_numpy(copy=False)

    if isinstance(cs, int):
        Cs_grid = np.logspace(-4, 4, cs)
    else:
        Cs_grid = np.asarray(list(cs), dtype=np.float64)

    if use_scaler is None:
        do_scale = not _is_binary_pm1(Xonly)
    else:
        do_scale = bool(use_scaler)

    lr_cv = CutlassLogisticCV(
        Cs=Cs_grid,
        solver=solver,
        cv=cv,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
        cv_rule=cv_rule,
        zero_clamp=zero_clamp,
        logic_polish=logic_polish,
        logic_scale=logic_scale,
        logic_target=logic_target,
        logic_maxk=logic_maxk,
        logic_rel_tol=logic_rel_tol,
        logic_plot=logic_plot,
        logic_plot_dir=logic_plot_dir,
        logic_Ks_plot=logic_Ks_plot,
        logic_k_policy=logic_k_policy,
        logic_smooth_k=logic_smooth_k,
        logic_firstmin_drop=logic_firstmin_drop,
        logic_firstmin_frac=logic_firstmin_frac,
        logic_intercept=logic_intercept,
        logic_m=logic_m,
        logic_m_frac=logic_m_frac,
    )

    steps: List[Tuple[str, object]] = []
    if do_scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("lr", lr_cv))

    pipe = Pipeline(steps)
    pipe.fit(Xonly.to_numpy(dtype=np.float64, copy=False), y)
    return pipe


def evaluate(pipe: Pipeline, X: pd.DataFrame, y_col: str = RESP):
    Xonly = X.drop(columns=[y_col])
    y = X[y_col].astype(int).to_numpy(copy=False)
    prob = pipe.predict_proba(Xonly.to_numpy(dtype=np.float64, copy=False))[:, 1]
    auc_val = roc_auc_score(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    fscore = 2 * rec * prec / (rec + prec + 1e-9)
    return prob, y.astype(bool), auc_val, fscore


def _polish_in_place_from_baseline(
    *,
    rect_base,              # fitted Pipeline on rectified features (logic_polish=False)
    rt_train,               # rectified training DataFrame (with RESP)
    intercept: str,         # "mean" | "mofk" | "maxj"
    k_policy: str,          # "global" | "premin"
    K: float,               # vote magnitude
    target: float | None,   # Youden J target or None
    logic_smooth_k: int = 7,
    logic_firstmin_drop: float = 0.05,
    logic_firstmin_frac: float = 0.5,
    Ks_plot = None,
    plot_dir: str | None = None,
    verbose: bool = False,
):
    """Run the built-in logical polishing routine on an already-fitted pipeline."""
    # Prepare arrays for _logical_polish
    Xtr = rt_train.drop(columns=[RESP]).to_numpy(dtype=np.float64, copy=False)
    ytr = rt_train[RESP].astype(int).to_numpy()

    # Grab the fitted LR step and set polishing knobs
    lr = rect_base.named_steps["lr"]
    lr.logic_intercept      = str(intercept).lower()
    lr.logic_k_policy       = str(k_policy).lower()
    lr.logic_smooth_k       = int(logic_smooth_k)
    lr.logic_firstmin_drop  = float(logic_firstmin_drop)
    lr.logic_firstmin_frac  = float(logic_firstmin_frac)

    # Snapshot original weights (baseline) from the *original fit*
    w0 = lr.coef_.ravel().copy()
    b0 = float(lr.intercept_[0])

    # Run the existing polisher from case_1_simple_script_gd_v8.py
    w_new, b_new, j_new, adopted, figs, diag = lr._logical_polish(
        X=Xtr, y=ytr, w=w0, b=b0,
        K=float(K),
        target=(None if target is None else float(target)),
        rel_tol=0.01,
        maxk=None,                        # scan all active features (or cap if you wish)
        make_plots=bool(plot_dir),
        Ks_plot=Ks_plot,
        plot_dir=(str(plot_dir) if plot_dir else None),
        verbose=bool(verbose),
    )

    # Attach diagnostics to the fitted object for downstream use
    lr.logic_figs_ = figs
    lr.logic_diag_ = diag

    # If adopted, overwrite the LR weights in place (turns baseline into the rule)
    if adopted:
        lr.coef_      = w_new.reshape(1, -1)
        lr.intercept_ = np.array([b_new], dtype=np.float64)

    return rect_base, diag, adopted

def save_sorted_and_feature_plots_for_run(
    *,
    figs_dir: Path,
    pipe_rect,
    pipe_raw,
    rt_train, rt_test,
    raw_train, raw_test,
    show_each: bool = False,
    threshold: float = 0.5
):
    """Write the standard probability and coefficient plots for a run."""

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Sorted probability plots (rectified) ---
    prob_tr_r, y_tr_r, _, _ = evaluate(pipe_rect, rt_train)
    prob_te_r, y_te_r, _, _ = evaluate(pipe_rect, rt_test)

    fig = plot_sorted(prob_tr_r, y_tr_r, "Training (Rectified)", h=threshold)
    maybe_show_or_save(fig, "Training (Rectified)", str(figs_dir), show_each)

    fig = plot_sorted(prob_te_r, y_te_r, "Test (Rectified)", h=threshold)
    maybe_show_or_save(fig, "Test (Rectified)", str(figs_dir), show_each)

    # --- 2) Sorted probability plots (raw baseline) ---
    prob_tr_raw, y_tr_raw, _, _ = evaluate(pipe_raw, raw_train)
    prob_te_raw, y_te_raw, _, _ = evaluate(pipe_raw, raw_test)

    fig = plot_sorted(prob_tr_raw, y_tr_raw, "Training (Raw, scaled)", h=threshold)
    maybe_show_or_save(fig, "Training (Raw, scaled)", str(figs_dir), show_each)

    fig = plot_sorted(prob_te_raw, y_te_raw, "Test (Raw, scaled)", h=threshold)
    maybe_show_or_save(fig, "Test (Raw, scaled)", str(figs_dir), show_each)

    # --- 3) Feature-selection / coefficient bar plots ---
    rect_feat_names = rt_train.drop(columns=[RESP]).columns.tolist()
    beta_rect = coefficient_series_only(pipe_rect, rect_feat_names, sort_for_plot=True)
    title_rect = "Coefficient Magnitudes (Rectified) - labeled true lags"
    highlights = LABELS
    fig = plot_coeff_bars(
        beta_rect,
        title_rect,
        highlights=highlights,
        annotate=bool(highlights),
    )
    maybe_show_or_save(fig, title_rect, str(figs_dir), show_each)

    raw_feat_names = raw_train.drop(columns=[RESP]).columns.tolist()
    beta_raw = coefficient_series_only(pipe_raw, raw_feat_names, sort_for_plot=False)
    title_raw = "Coefficient Magnitudes (Raw) - labeled true lags"
    fig = plot_coeff_bars(beta_raw, title_raw, highlights=highlights, annotate=False)
    maybe_show_or_save(fig, title_raw, str(figs_dir), show_each)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _youden_from_probs(probs: np.ndarray, y_bool: np.ndarray, thr: float = 0.5) -> float:
    return youdenJ(y_bool, probs >= thr)

def _count_nz(pipe) -> int:
    lr = pipe.named_steps["lr"]
    return int(np.count_nonzero(lr.coef_.ravel()))

def _extract_logic_diag(pipe) -> dict:
    lr = pipe.named_steps["lr"]
    return getattr(lr, "logic_diag_", {}) or {}

def _fit_eval_sklearn(rt_train, rt_test, raw_train, raw_test, *, cv: int, cs: int = 25):
    """
    Fit scikit-learn baselines using the fast v4 script:
      - Rectified view (no scaler):  skcase.build_model(..., use_scaler=False)
      - Raw view (with scaler):      skcase.build_model(..., use_scaler=True)

    Returns:
      dict with: t_rect_skl, t_raw_skl, auc_te_rect_skl, j_te_rect_skl,
                 f1max_te_rect_skl, auc_te_raw_skl
    """
    # --- Rectified (sklearn) ---
    t0 = time.perf_counter()
    mdl_rect = skcase.build_model(rt_train, cv=cv, cs=cs, use_scaler=False)  # v4 API
    t_rect_skl = time.perf_counter() - t0

    prob_r_te, y_r_te, auc_r_te, f1_r_te = skcase.evaluate(mdl_rect, rt_test)  # v4 API
    j_te_rect_skl = youdenJ(y_r_te, (prob_r_te >= 0.5))
    f1max_te_rect_skl = float(np.nanmax(f1_r_te))
    auc_te_rect_skl = float(auc_r_te)

    # --- Raw (sklearn) ---
    t1 = time.perf_counter()
    mdl_raw = skcase.build_model(raw_train, cv=cv, cs=cs, use_scaler=True)   # v4 API
    t_raw_skl = time.perf_counter() - t1

    prob_w_te, y_w_te, auc_w_te, _ = skcase.evaluate(mdl_raw, raw_test)     # v4 API
    auc_te_raw_skl = float(auc_w_te)

    return dict(
        t_rect_skl=t_rect_skl, t_raw_skl=t_raw_skl,
        auc_te_rect_skl=auc_te_rect_skl, j_te_rect_skl=j_te_rect_skl,
        f1max_te_rect_skl=f1max_te_rect_skl, auc_te_raw_skl=auc_te_raw_skl
    )

def _save_roc_csv(path: Path, fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> None:
    """Save ROC arrays to CSV (fpr,tpr,threshold)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    df.to_csv(path, index=False)

def _roc_from_probs(y_true: np.ndarray, y_score: np.ndarray):
    """Return (fpr, tpr, thr, auc_val) from truth and score arrays."""
    fpr, tpr, thr = roc_curve(y_true.astype(int), y_score.astype(float))
    return fpr, tpr, thr, auc(fpr, tpr)

def _roc_from_pipe(pipe, df):
    """
    Use your existing evaluate(...) to get (probs, y) and compute ROC.
    Works for both rectified and raw pipelines.
    """
    probs, y_bool, _, _ = evaluate(pipe, df)
    return _roc_from_probs(y_bool, probs)

def save_roc_curves_for_run(
    *,
    figs_dir: Path,
    run_id: str,
    # your fitted models/pipelines
    pipe_rule,            # rectified pipeline *after* polishing (rule possibly adopted in-place)
    pipe_rect_baseline,   # rectified baseline pipeline *before* polishing (if available); else pass None and provide arrays below
    pipe_raw_baseline,    # raw (scaled) baseline pipeline
    # per-run datasets
    rt_train, rt_test,    # rectified train/test DataFrames (RESP present)
    raw_train, raw_test,  # raw train/test DataFrames (RESP present)
    # (optional) baseline rectified arrays captured before in-place polish
    rect_base_yte: np.ndarray | None = None,
    rect_base_probte: np.ndarray | None = None,
    # include sklearn curves?
    include_sklearn: bool = True,
    sklearn_cv: int = 3,
    skcase_module = None  # pass the imported `skcase` module if available; else we skip sklearn curves
):
    """
    Create a single ROC figure overlaying:
      - Rule model (rectified, after polish)
      - Rectified logistic (baseline)
      - Raw logistic (baseline)
      - Scikit-learn (rectified + raw), if requested and skcase_module is provided

    Also save per-model ROC CSVs (fpr,tpr,threshold) into figs_dir.
    """
    figs_dir = Path(figs_dir); figs_dir.mkdir(parents=True, exist_ok=True)

    # --- Rule model (rectified after polish)
    fpr_rule, tpr_rule, thr_rule, auc_rule = _roc_from_pipe(pipe_rule, rt_test)
    _save_roc_csv(figs_dir / "roc_rule.csv", fpr_rule, tpr_rule, thr_rule)

    # --- Rectified logistic baseline
    if rect_base_yte is not None and rect_base_probte is not None:
        fpr_rect, tpr_rect, thr_rect, auc_rect = _roc_from_probs(rect_base_yte, rect_base_probte)
    elif pipe_rect_baseline is not None:
        fpr_rect, tpr_rect, thr_rect, auc_rect = _roc_from_pipe(pipe_rect_baseline, rt_test)
    else:
        # As a fallback, recompute from the current (possibly rule-updated) pipe; not ideal
        fpr_rect, tpr_rect, thr_rect, auc_rect = _roc_from_pipe(pipe_rule, rt_test)
    _save_roc_csv(figs_dir / "roc_rectified_baseline.csv", fpr_rect, tpr_rect, thr_rect)

    # --- Raw logistic baseline
    fpr_raw, tpr_raw, thr_raw, auc_raw = _roc_from_pipe(pipe_raw_baseline, raw_test)
    _save_roc_csv(figs_dir / "roc_raw_baseline.csv", fpr_raw, tpr_raw, thr_raw)

    # --- Optional: scikit-learn curves (rectified + raw)
    have_sklearn = include_sklearn and (skcase_module is not None)
    if have_sklearn:
        # Rectified (sklearn v4)
        mdl_rect = skcase_module.build_model(rt_train, cv=sklearn_cv, use_scaler=False)
        prob_skr, y_skr, _, _ = skcase_module.evaluate(mdl_rect, rt_test)
        fpr_skr, tpr_skr, thr_skr, auc_skr = _roc_from_probs(y_skr, prob_skr)
        _save_roc_csv(figs_dir / "roc_sklearn_rectified.csv", fpr_skr, tpr_skr, thr_skr)

        # Raw (sklearn v4)
        mdl_raw = skcase_module.build_model(raw_train, cv=sklearn_cv, use_scaler=True)
        prob_skw, y_skw, _, _ = skcase_module.evaluate(mdl_raw, raw_test)
        fpr_skw, tpr_skw, thr_skw, auc_skw = _roc_from_probs(y_skw, prob_skw)
        _save_roc_csv(figs_dir / "roc_sklearn_raw.csv", fpr_skw, tpr_skw, thr_skw)
    else:
        fpr_skr = tpr_skr = thr_skr = auc_skr = None
        fpr_skw = tpr_skw = thr_skw = auc_skw = None

    # --- Combined figure
    plt.figure(figsize=(6.6, 5.4))
    if have_sklearn:
        plt.plot(fpr_skr, tpr_skr, linestyle="--", label=f"sklearn Rectified (AUC={auc_skr:.4f})")
        plt.plot(fpr_skw, tpr_skw, linestyle="--", label=f"sklearn Raw (AUC={auc_skw:.4f})")
    plt.plot(fpr_raw,  tpr_raw,  label=f"Raw L1 (AUC={auc_raw:.4f})")
    plt.plot(fpr_rect, tpr_rect, label=f"Rectified L1 (AUC={auc_rect:.4f})")
    plt.plot(fpr_rule, tpr_rule, label=f"Rule (AUC={auc_rule:.4f})")

    plt.plot([0,1],[0,1], 'k--', linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {run_id}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figs_dir / "ROC_All_Models.png", dpi=200)
    plt.close()

    # --- Optional: also store a single combined CSV for convenience
    comb = []
    comb.append(pd.DataFrame({"model":"rule", "fpr":fpr_rule, "tpr":tpr_rule, "threshold":thr_rule}))
    comb.append(pd.DataFrame({"model":"rectified_l1", "fpr":fpr_rect, "tpr":tpr_rect, "threshold":thr_rect}))
    comb.append(pd.DataFrame({"model":"raw_l1", "fpr":fpr_raw, "tpr":tpr_raw, "threshold":thr_raw}))
    if have_sklearn:
        comb.append(pd.DataFrame({"model":"sklearn_rectified", "fpr":fpr_skr, "tpr":tpr_skr, "threshold":thr_skr}))
        comb.append(pd.DataFrame({"model":"sklearn_raw", "fpr":fpr_skw, "tpr":tpr_skw, "threshold":thr_skw}))
    pd.concat(comb, ignore_index=True).to_csv(figs_dir / "roc_all_models.csv", index=False)

# ---------- one experimental run ----------
def run_one(
    *,
    outdir: Path,
    seed: int,
    tr_frac: Optional[float],
    intercept: str,
    k_policy: str,
    K: float,
    target: float | None,
    dataset_csv: Optional[Path],
    train_csv: Optional[Path],
    test_csv: Optional[Path],
    sdfilter: float = 3.0,
    snap: float = 0.001,
    cv: int = 3,
    cs: int = 25,
    solver: str = "hybrid",
    zero_clamp: float = 1e-3,
    save_artifacts: bool = True,
):
    # 1) Load (or reuse) provided dataset(s)
    print("  Loading dataset(s) ", end=""); timestamp()
    t0 = time.perf_counter()
    train, test, dataset_tag, actual_tr_frac, n_examples, _explicit_split = _prepare_train_test(
        seed=seed,
        dataset_csv=dataset_csv,
        train_csv=train_csv,
        test_csv=test_csv,
        train_frac=tr_frac,
    )
    t_load = time.perf_counter() - t0

    print(
        f"        -> {len(train)} train / {len(test)} test (total={n_examples}, frac={actual_tr_frac:.4f})",
        flush=True,
    )

    frac_for_id = actual_tr_frac if not math.isnan(actual_tr_frac) else None
    frac_label = "NA" if frac_for_id is None else f"{int(round(100 * frac_for_id))}"
    target_label = "NA" if target is None else int(round(100 * target))
    run_id = f"{dataset_tag}_seed{seed}_N{n_examples}_tr{frac_label}_{intercept}_{k_policy}_K{K}_T{target_label}"
    rdir = _ensure_dir(outdir / run_id)
    (rdir / "figs").mkdir(exist_ok=True)

    (rdir / "raw_splits").mkdir(parents=True, exist_ok=True)
    train.to_csv(rdir / "raw_splits" / "train.csv", index=False)
    test.to_csv(rdir / "raw_splits" / "test.csv", index=False)

    # 2) Rectify train/test with the SAME limits
    print("  Rectifying the data ", end=""); timestamp()
    groups = organize(train)
    rt_train, limits, rectifier = rectify_fast(
        train, groups, rectifier=None, sdfilter=sdfilter, snap=snap
    )
    rt_test, _, _ = rectify_fast(
        test, groups, rectifier=rectifier, sdfilter=sdfilter, snap=snap
    )

    # Raw views (keep response)
    raw_train = train[[c for c in train.columns if c not in EXCL]]
    raw_test  = test [[c for c in test.columns  if c not in EXCL]]

    # 3) ONE rectified fit (NO logical polish)
    print("  [1/4] Building the model with the rectified data ", end=""); timestamp()
    t1 = time.perf_counter()
    rect_base = build_model(rt_train, cv=cv, cs=cs, use_scaler=False,
                            solver=solver, cv_rule="1se", zero_clamp=zero_clamp,
                            logic_polish=False, verbose=False)
    t_rect_base = time.perf_counter() - t1

    # Baseline rectified metrics BEFORE polishing
    prob_tr_b, y_tr_b, auc_tr_b, f1_tr_b = evaluate(rect_base, rt_train)
    prob_te_b, y_te_b, auc_te_b, f1_te_b = evaluate(rect_base, rt_test)
    j_tr_b = _youden_from_probs(prob_tr_b, y_tr_b)
    j_te_b = _youden_from_probs(prob_te_b, y_te_b)
    k_rect_base = _count_nz(rect_base)

    # 4) Run logic polishing IN PLACE (no refit)
    t2 = time.perf_counter()
    # Reuse the helper you already added earlier (present in this file)
    print("        Run logic polishing ", end=""); timestamp()
    rect_rule, diag, adopted = _polish_in_place_from_baseline(
        rect_base=rect_base,
        rt_train=rt_train,
        intercept=intercept,
        k_policy=k_policy,
        K=float(K),
        target=(None if target is None else float(target)),
        logic_smooth_k=7, logic_firstmin_drop=0.05, logic_firstmin_frac=0.5,
        Ks_plot=None, plot_dir=str(rdir / "figs"), verbose=False,
    )
    t_rect_rule = time.perf_counter() - t2

    # Metrics after (possible) adoption of the rule
    prob_tr_r, y_tr_r, auc_tr_r, f1_tr_r = evaluate(rect_rule, rt_train)
    prob_te_r, y_te_r, auc_te_r, f1_te_r = evaluate(rect_rule, rt_test)
    j_tr_r  = _youden_from_probs(prob_tr_r, y_tr_r)
    j_te_r  = _youden_from_probs(prob_te_r, y_te_r)
    k_rect_rule = _count_nz(rect_rule)

    # 5) Raw baseline (your implementation)
    t3 = time.perf_counter()
    print("  [2/4] Building the model with the raw data ", end=""); timestamp()
    raw_base = build_model(raw_train, cv=cv, cs=cs, use_scaler=True,
                           solver=solver, cv_rule="1se", zero_clamp=zero_clamp,
                           logic_polish=False, verbose=False)
    t_raw_base = time.perf_counter() - t3
    _, _, auc_tr_raw, _ = evaluate(raw_base, raw_train)
    _, _, auc_te_raw, _ = evaluate(raw_base, raw_test)

    # --- 5b) sklearn baselines via EXTERNAL CV (no warm-start), apples-to-apples ---

    # Rectified (+/-1) - no scaler
    t_skr0 = time.perf_counter()
    print("  [3/4] Building the sklearn model with the rectified data ", end=""); timestamp()
    sk_rect_pipe, sk_rect_info = sklearn_build_rectified_external_cv(
        rt_train,
        cv=cv, cs=cs, c_lo=-4, c_hi=4,         # match gd_v8 grid if desired
        cv_rule="1se",                         # or "min" to match your run
        random_state=seed, solver="saga",
        tol=1e-3, max_iter=2000
    )
    t_rect_skl_ext = time.perf_counter() - t_skr0
    prob_skr_tr, y_skr_tr, auc_skr_tr, _ = evaluate(sk_rect_pipe, rt_train)
    prob_skr_te, y_skr_te, auc_skr_te, _ = evaluate(sk_rect_pipe, rt_test)
    j_skr_tr = _youden_from_probs(prob_skr_tr, y_skr_tr)
    j_skr_te = _youden_from_probs(prob_skr_te, y_skr_te)
    
    # Raw (scaled) - with StandardScaler
    t_skw0 = time.perf_counter()
    print("  [4/4] Building the sklearn model with the raw data ", end=""); timestamp()
    sk_raw_pipe, sk_raw_info = sklearn_build_raw_external_cv(
        raw_train,
        cv=cv, cs=cs, c_lo=-4, c_hi=4,
        cv_rule="1se",
        random_state=seed, solver="saga",
        tol=1e-3, max_iter=2000
    )
    t_raw_skl_ext = time.perf_counter() - t_skw0
    prob_skw_te, y_skw_te, auc_skw_te, _ = evaluate(sk_raw_pipe, raw_test)
    prob_skw_tr, y_skw_tr, auc_skw_tr, _ = evaluate(sk_raw_pipe, raw_train)
    
    # Save sklearn ROC CSVs using the same helpers we already have
    fpr_skr, tpr_skr, thr_skr, _ = _roc_from_probs(y_skr_te, prob_skr_te)
    _save_roc_csv((rdir / "figs" / "roc_sklearn_rectified_extcv.csv"), fpr_skr, tpr_skr, thr_skr)
    
    fpr_skw, tpr_skw, thr_skw, _ = _roc_from_probs(y_skw_te, prob_skw_te)
    _save_roc_csv((rdir / "figs" / "roc_sklearn_raw_extcv.csv"), fpr_skw, tpr_skw, thr_skw)


    # Diagnostics from polisher
    adopted_flag   = bool(diag.get("adopted", False))
    k_chosen       = int(diag.get("k_chosen", 0))
    J_chosen_train = float(diag.get("J_chosen", float("nan")))
    J0_train       = float(diag.get("J0", float("nan")))
    K_used         = float(diag.get("K_used", float("nan")))
    policy         = str(diag.get("policy", ""))
    k_first_min    = int(diag.get("k_first_min", 0))

    # Save artifacts (optional)
    # Always persist the datasets used for this run so downstream analysis can
    # reproduce results irrespective of the artifact flag.
    rt_train.to_csv(rdir / "rt_train.csv", index=False)
    rt_test.to_csv(rdir / "rt_test.csv", index=False)
    raw_train.to_csv(rdir / "raw_train.csv", index=False)
    raw_test.to_csv(rdir / "raw_test.csv", index=False)

    if save_artifacts:
        with open(rdir / "logic_diag.json", "w") as f:
            json.dump(diag, f, indent=2)

    # Per-run plots (rectified rule + raw baseline)
    save_sorted_and_feature_plots_for_run(
        figs_dir=rdir / "figs",
        pipe_rect=rect_rule,
        pipe_raw=raw_base,
        rt_train=rt_train, rt_test=rt_test,
        raw_train=raw_train, raw_test=raw_test,
        show_each=False, threshold=0.5
    )
    
    # --- Row #1: your original pipeline ---
    row_original = dict(
        run_id=run_id, seed=seed, n_examples=n_examples, train_frac=actual_tr_frac,
        intercept=intercept, k_policy=k_policy, K=K, target=(None if target is None else target),
        t_gen=t_load, t_rect_base=t_rect_base, t_rect_rule=t_rect_rule, t_raw_base=t_raw_base,
        auc_tr_rect_base=auc_tr_b, j_tr_rect_base=j_tr_b, f1max_tr_rect_base=float(np.nanmax(f1_tr_b)),
        auc_tr_rect_rule=auc_tr_r, j_tr_rect_rule=j_tr_r, f1max_tr_rect_rule=float(np.nanmax(f1_tr_r)),
        auc_tr_raw=auc_tr_raw,
        auc_te_rect_base=auc_te_b, j_te_rect_base=j_te_b, f1max_te_rect_base=float(np.nanmax(f1_te_b)),
        auc_te_rect_rule=auc_te_r, j_te_rect_rule=j_te_r, f1max_te_rect_rule=float(np.nanmax(f1_te_r)),
        auc_te_raw=auc_te_raw,
        adopted=adopted_flag, k_chosen=k_chosen, k_first_min=k_first_min,
        k_rect_base=k_rect_base, k_rect_rule=k_rect_rule,
        J_chosen_train=J_chosen_train, J0_train=J0_train,
        policy_trained=policy, K_used=K_used
    )

    # --- Row #2: scikit-learn baseline on the same data ---
    
    # skl = _fit_eval_sklearn(rt_train, rt_test, raw_train, raw_test, cv=cv, cs=cs)
    # row_sklearn = dict(
    #     run_id=run_id + "__sklearn", seed=seed, n_examples=n_examples, train_frac=tr_frac,
    #     # leave these BLANK per your request
    #     intercept="", k_policy="", K="", target="", adopted="", k_chosen="", k_first_min="",
    #     k_rect_base="", k_rect_rule="", K_used="",
    #     # timings from scikit runs
    #     t_gen=t_gen,  # same generation time
    #     t_rect_base=skl["t_rect_skl"], t_rect_rule=np.nan, t_raw_base=skl["t_raw_skl"],
    #     # scikit metrics
    #     auc_te_rect_base=skl["auc_te_rect_skl"],
    #     j_te_rect_base=skl["j_te_rect_skl"],
    #     f1max_te_rect_base=skl["f1max_te_rect_skl"],
    #     # no rule step under sklearn - leave rule metrics blank/NaN
    #     auc_te_rect_rule=np.nan, j_te_rect_rule=np.nan, f1max_te_rect_rule=np.nan,
    #     auc_te_raw=skl["auc_te_raw_skl"],
    #     # adoption fields blank; policy marks this as scikit result
    #     J_chosen_train=np.nan, J0_train=np.nan,
    #     policy_trained="sklearn"
    # )
    
    row_sklearn_ext = dict(
        run_id=run_id + "__sklearn_extcv",
        seed=seed, n_examples=n_examples, train_frac=actual_tr_frac,
        # leave rule fields blank
        intercept="", k_policy="", K="", target="",
        adopted="", k_chosen="", k_first_min="", k_rect_base="", k_rect_rule="", K_used="",
        # timings
        t_gen=t_load,
        t_rect_base=t_rect_skl_ext, t_rect_rule=np.nan, t_raw_base=t_raw_skl_ext,
        # metrics
        auc_tr_rect_base=float(auc_skr_tr),
        j_tr_rect_base=float(j_skr_tr),
        f1max_tr_rect_base=np.nan,     
        auc_tr_rect_rule=np.nan, j_tr_rect_rule=np.nan, f1max_tr_rect_rule=np.nan,
        auc_tr_raw=float(auc_skw_tr),
        auc_te_rect_base=float(auc_skr_te),
        j_te_rect_base=float(j_skr_te),
        f1max_te_rect_base=np.nan,     
        auc_te_rect_rule=np.nan, j_te_rect_rule=np.nan, f1max_te_rect_rule=np.nan,
        auc_te_raw=float(auc_skw_te),
        # polish-specific diagnostics not applicable
        J_chosen_train=np.nan, J0_train=np.nan,
        policy_trained="sklearn_extcv"
    )
    
    # If run_one returns a list, append this alongside the original row.

    # We already have these from earlier in run_one:
    # prob_te_b, y_te_b  -> rectified baseline test probs/labels
    # rect_rule          -> rectified pipeline after polish (rule possibly adopted in-place)
    # raw_base           -> raw (scaled) baseline pipeline
    # rt_train, rt_test, raw_train, raw_test, run_id, rdir, etc.
    
    # If scikit helper module was imported earlier, pass it in as skcase; else pass None.
    try:
        sk_mod = skcase  # if you imported case_1_simple_script_scikit.py earlier in the driver
    except NameError:
        sk_mod = None
    
    save_roc_curves_for_run(
        figs_dir=rdir / "figs",
        run_id=run_id,
        pipe_rule=rect_rule,
        pipe_rect_baseline=None,              # we use the stored arrays so this can be None
        pipe_raw_baseline=raw_base,
        rt_train=rt_train, rt_test=rt_test,
        raw_train=raw_train, raw_test=raw_test,
        rect_base_yte=y_te_b, rect_base_probte=prob_te_b,
        include_sklearn=True,                 # set False if you want to skip sklearn
        sklearn_cv=cv,
        skcase_module=sk_mod
    )

    # Return BOTH rows
    return [row_original, row_sklearn_ext]

# ---------- grid runner ----------
def main():
    # Default configuration mirrors experiment_driver_v5 but without synthetic generation.
    default_outdir = "runs_new"
    default_seeds = [1234]
    default_dataset_csv = HERE / "sample_data" / "leukemia_gene_expression.csv"
    if not default_dataset_csv.exists():
        default_dataset_csv = None
    default_tr_fracs = [0.25]
    default_Ks = [10.0]
    default_k_policies = ["global"]
    default_intercepts = ["mofk"]  # "mean","mofk","maxj"
    default_targets: List[float] = []
    default_cv = 3
    default_cs = 25
    default_solver = "hybrid"
    default_sdfilter = 3.0
    default_snap = 0.001
    default_zero_clamp = 1e-3
    default_no_artifacts = True

    ap = argparse.ArgumentParser(description="Run CUTLASS experiments using existing CSV splits.")
    ap.add_argument("--outdir", type=str, default=default_outdir)
    ap.add_argument("--seeds", type=int, nargs="+", default=default_seeds)
    ap.add_argument("--dataset-csv", type=str, default=None,
                    help="Single dataset CSV to be split into train/test.")
    ap.add_argument("--train-csv", type=str, default=None, help="Training CSV when providing explicit splits.")
    ap.add_argument("--test-csv", type=str, default=None, help="Test CSV when providing explicit splits.")
    ap.add_argument("--tr-fracs", type=float, nargs="+", default=None,
                    help="Training fractions used when splitting a single dataset.")
    ap.add_argument("--targets", type=float, nargs="*", default=default_targets,
                    help="Optional list of Youden-J targets.")
    ap.add_argument("--Ks", type=float, nargs="+", default=default_Ks)
    ap.add_argument("--k-policies", type=str, nargs="+", default=default_k_policies, choices=["global", "premin"])
    ap.add_argument("--intercepts", type=str, nargs="+", default=default_intercepts, choices=["mean", "mofk", "maxj"])
    ap.add_argument("--cv", type=int, default=default_cv)
    ap.add_argument("--cs", type=int, default=default_cs)
    ap.add_argument("--solver", type=str, default=default_solver)
    ap.add_argument("--sdfilter", type=float, default=default_sdfilter)
    ap.add_argument("--snap", type=float, default=default_snap)
    ap.add_argument("--zero-clamp", type=float, default=default_zero_clamp)
    ap.add_argument("--no-artifacts", action="store_true", default=default_no_artifacts,
                    help="Skip saving intermediate artifacts (default).")
    ap.add_argument("--artifacts", dest="no_artifacts", action="store_false",
                    help="Save intermediate artifacts.")
    args = ap.parse_args()

    def _maybe_path(value: Optional[str]) -> Optional[Path]:
        if value is None:
            return None
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped.lower() == "none":
            return None
        return Path(stripped)

    outdir = _ensure_dir(Path(args.outdir))
    train_csv = _maybe_path(args.train_csv)
    test_csv = _maybe_path(args.test_csv)
    dataset_csv = _maybe_path(args.dataset_csv)
    if dataset_csv is None and default_dataset_csv and train_csv is None and test_csv is None:
        dataset_csv = default_dataset_csv
    seeds = [int(s) for s in args.seeds]
    Ks = [float(k) for k in args.Ks]
    k_policies = [str(kp) for kp in args.k_policies]
    intercepts = [str(ip) for ip in args.intercepts]
    if args.tr_fracs is None:
        tr_fracs = list(default_tr_fracs)
        tr_fracs_explicit = False
    else:
        tr_fracs = [float(frac) for frac in args.tr_fracs]
        tr_fracs_explicit = True
    target_values = [float(t) for t in args.targets] if args.targets else []
    save_artifacts = not args.no_artifacts
    sdfilter = float(args.sdfilter)
    snap = float(args.snap)
    zero_clamp = float(args.zero_clamp)
    cv = int(args.cv)
    cs = int(args.cs)
    solver = str(args.solver)

    if dataset_csv and not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    if train_csv and not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if test_csv and not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    use_dataset = dataset_csv is not None
    use_explicit_split = (train_csv is not None) or (test_csv is not None)
    if use_dataset and use_explicit_split:
        raise ValueError("Specify either --dataset-csv or --train-csv/--test-csv, but not both.")
    if not use_dataset and not use_explicit_split:
        raise ValueError("Provide --dataset-csv or both --train-csv and --test-csv.")
    if use_explicit_split and (train_csv is None or test_csv is None):
        raise ValueError("Both --train-csv and --test-csv are required when providing explicit splits.")

    if use_dataset:
        if not tr_fracs:
            raise ValueError("At least one training fraction is required when using --dataset-csv.")
        tr_frac_iterable: List[Optional[float]] = tr_fracs
    else:
        if tr_fracs_explicit and tr_fracs:
            print("[warning] Training fractions ignored because explicit train/test CSVs were provided.", flush=True)
        tr_frac_iterable = [None]

    target_iterable: List[Optional[float]] = target_values if target_values else [None]

    summary_path = outdir / "summary.csv"
    
    rows = []
    for (seed, trf, K, ip, kp, tgt) in itertools.product(
        seeds, tr_frac_iterable, Ks, intercepts, k_policies, target_iterable
    ):
        tr_desc = "explicit" if trf is None else f"{trf:.2f}"
        tgt_desc = "None" if tgt is None else f"{tgt:.3f}"
        print(
            f"*** Starting iteration for seed={seed}, trf={tr_desc}, K={K}, intercept={ip}, "
            f"k_policy={kp}, target={tgt_desc} ",
            end=""
        ); timestamp()
        row_pair = run_one(
            outdir=outdir,
            seed=int(seed), tr_frac=(None if trf is None else float(trf)),
            intercept=str(ip), k_policy=str(kp),
            K=float(K), target=(None if tgt is None else float(tgt)),
            dataset_csv=dataset_csv, train_csv=train_csv, test_csv=test_csv,
            sdfilter=sdfilter, snap=snap,
            cv=cv, cs=cs, solver=solver,
            zero_clamp=zero_clamp,
            save_artifacts=save_artifacts
        )
        rows.extend(row_pair)   # <<- now we extend with [original, sklearn]
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"[done] wrote {summary_path} with {len(rows)} rows")

if __name__ == "__main__":
    main()


