# Vignette 3: Batched Experiments and Artifact Export

This vignette condenses the functionality of `experiment_driver_v5.py` into a
few reusable helpers built on top of the packaged API.  It demonstrates how to

1. generate multiple train/test splits,
2. evaluate alternative logic-polish targets and intercept policies,
3. capture metrics and diagnostics in a tidy results table, and
4. persist rectifier limits + fitted weights for deployment.

```python
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cutlass import CutlassClassifier
from cutlass.metrics import calculate_youden_j, roc_auc_score
from cutlass.serialization import save_classifier_npz, save_limits_json


@dataclass
class ExperimentConfig:
    seed: int
    train_frac: float
    logic_target: float | None
    logic_k_policy: str
    logic_intercept: str
    logic_scale: float


def prepare_split(df: pd.DataFrame, resp: str, *, seed: int, frac: float):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(frac * len(df))
    tr_idx, te_idx = idx[:cut], idx[cut:]
    return df.iloc[tr_idx], df.iloc[te_idx]


def run_single_experiment(df: pd.DataFrame, resp: str, cfg: ExperimentConfig):
    train_df, test_df = prepare_split(df, resp, seed=cfg.seed, frac=cfg.train_frac)
    X_tr, y_tr = train_df.drop(columns=[resp]), train_df[resp]
    X_te, y_te = test_df.drop(columns=[resp]), test_df[resp]

    clf = CutlassClassifier(
        rectify=True,
        Cs=21,
        solver="cd",
        cv=5,
        random_state=cfg.seed,
        logic_polish=True,
        logic_scale=cfg.logic_scale,
        logic_target=cfg.logic_target,
        logic_k_policy=cfg.logic_k_policy,
        logic_intercept=cfg.logic_intercept,
        logic_plot=False,
    )
    clf.fit(X_tr, y_tr)

    prob_tr = clf.predict_proba(X_tr)[:, 1]
    prob_te = clf.predict_proba(X_te)[:, 1]

    yhat_tr = (prob_tr >= 0.5).astype(int)
    yhat_te = (prob_te >= 0.5).astype(int)

    diag = clf.classifier_.logic_diag_

    metrics = {
        "seed": cfg.seed,
        "train_frac": cfg.train_frac,
        "logic_target": cfg.logic_target,
        "logic_k_policy": cfg.logic_k_policy,
        "logic_intercept": cfg.logic_intercept,
        "logic_scale": cfg.logic_scale,
        "adopted": diag.get("adopted", False),
        "k_chosen": diag.get("k_chosen"),
        "J_train": calculate_youden_j(y_tr, yhat_tr),
        "J_test": calculate_youden_j(y_te, yhat_te),
        "auc_train": roc_auc_score(y_tr, prob_tr),
        "auc_test": roc_auc_score(y_te, prob_te),
    }

    artifacts = {
        "limits": clf.limits_,
        "logic_diag": diag,
    }
    return metrics, clf, artifacts


# --- Example driver ----------------------------------------------------------

resp = "INDC"
raw_df = pd.read_csv("path/to/dataset.csv")  # Replace with project data.

configs = [
    ExperimentConfig(seed=2025, train_frac=0.6, logic_target=0.95,
                     logic_k_policy="global", logic_intercept="mean", logic_scale=10.0),
    ExperimentConfig(seed=2025, train_frac=0.6, logic_target=0.97,
                     logic_k_policy="premin", logic_intercept="maxj", logic_scale=12.0),
    ExperimentConfig(seed=3141, train_frac=0.7, logic_target=None,
                     logic_k_policy="global", logic_intercept="mofk", logic_scale=8.0),
]

rows = []
outdir = Path("runs/vignette_demo")
outdir.mkdir(parents=True, exist_ok=True)

for i, cfg in enumerate(configs, start=1):
    metrics, model, artifacts = run_single_experiment(raw_df, resp, cfg)
    rows.append(metrics)

    run_dir = outdir / f"run_{i:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist rectifier limits + L1 coefficients.
    save_limits_json(model.limits_, run_dir / "limits.json")
    save_classifier_npz(model, model.feature_names_, run_dir / "classifier.npz")

    # Diagnostics collected during logic polishing.
    (run_dir / "logic_diag.json").write_text(json.dumps(artifacts["logic_diag"], indent=2))

results = pd.DataFrame(rows)
results.to_csv(outdir / "summary.csv", index=False)
print(results)
```

The helper pattern retains the modularity of the original experiment driver
while depending solely on the published package.  You can expand on this by
adding joblib-based parallelism, plotting hooks, or target grids just as in the
prototype script.
