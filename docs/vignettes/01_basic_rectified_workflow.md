# Vignette 1: Rectified L1 Logistic Workflow

This vignette mirrors the “rectify → fit → evaluate” path in the original
experiment scripts using the new `cutlass` package.  We start with a tabular
data frame, apply the built‑in rectifier that learns critical ranges from the
positive class, and fit a cross‑validated L1 logistic model.

```python
import numpy as np
import pandas as pd
from cutlass import CutlassClassifier, calculate_youden_j, roc_auc_score

# --- 1. Prepare a toy binary dataset ----------------------------------------
rng = np.random.default_rng(42)
n = 1000
X = pd.DataFrame(
    {
        "V1_t0": rng.normal(loc=0.0, scale=1.0, size=n),
        "V1_t1": rng.normal(loc=0.1, scale=1.2, size=n),
        "V2_t0": rng.uniform(-2.0, 2.0, size=n),
        "V2_t1": rng.uniform(-2.0, 2.0, size=n),
    }
)

# Target depends on critical ranges similar to the paper's construction.
y = (
    ((X["V1_t0"] > 0.6) & (X["V2_t1"] > 0.7))
    | ((X["V1_t1"] < -0.4) & (X["V2_t0"] < -0.5))
).astype(int)

# Combine into a modelling frame.
df = X.assign(INDC=y)

# --- 2. Fit the CUTLASS classifier -------------------------------------------
clf = CutlassClassifier(
    rectify=True,
    Cs=21,
    solver="cd",
    cv=5,
    logic_polish=False,  # keep plain L1 model for now
)
clf.fit(df.drop(columns=["INDC"]), df["INDC"])

# --- 3. Inspect results ------------------------------------------------------
prob = clf.predict_proba(df.drop(columns=["INDC"]))[:, 1]
pred = (prob >= 0.5).astype(int)

print("ROC AUC:", roc_auc_score(df["INDC"], prob))
print("Youden J:", calculate_youden_j(df["INDC"], pred))
print("Selected features:", np.flatnonzero(clf.classifier_.coef_))
print("Rectifier limits:", clf.limits_)
```

Key points:

- The `CutlassClassifier` mirrors the old `build_model` helper.
- Rectification happens internally and the learned limits are exposed via
  `clf.limits_` for deployment or inspection.
- Metrics such as Youden’s J and ROC AUC come from `cutlass.metrics`.
