# CUTLASS

CUTLASS (Critical-range rectified LASSO) packages the workflow developed in the
project scripts into a reusable, publishable Python library.  It exposes a
scikit-learn inspired estimator that rectifies the input space into
\{-1, +1\} indicators, trains an L1-penalised logistic model with an efficient
coordinate-descent solver, and optionally compresses the model into a logical
rule without any dependence on scikit-learn itself.

## Features

- **Rectifier transformer** that infers critical ranges from the positive class
  and binarises features into \{-1, +1\}.
- **Cross-validated L1 logistic model** with warm-started coordinate descent
  and optional FISTA solver.
- **Logical compression** step mirroring the research code (top-k votes with
  fixed magnitude `K` and several intercept policies).
- **Serialization helpers** to persist rectifier limits and fitted models.
- Lightweight implementation that depends only on NumPy and pandas
  (matplotlib is optional for plotting diagnostics).

## Installation

```bash
pip install cutlass
```

The plotting utilities used by the logical compression step are optional.  To
enable them, install the `plots` extra:

```bash
pip install cutlass[plots]
```

## Quick start

```python
import pandas as pd
from cutlass import CutlassClassifier

# toy binary dataset
df = pd.DataFrame(
    {
        "feat_a": [0.1, 0.3, 0.7, 0.9],
        "feat_b": [10, 13, 8, 5],
        "INDC": [0, 0, 1, 1],
    }
)

X = df.drop(columns=["INDC"])
y = df["INDC"]

clf = CutlassClassifier(
    rectify=True,
    Cs=15,
    solver="cd",
    cv=3,
    logic_polish=True,
    logic_scale=10.0,
)
clf.fit(X, y)
print(clf.predict_proba(X))
print("limits:", clf.limits_)
```

## Vignettes

Additional step-by-step guides live under `docs/vignettes/`:

- `01_basic_rectified_workflow.md` – reproduce the baseline rectified L1 fit.
- `02_logical_polish.md` – enable logic polishing and interpret diagnostics.
- `03_batch_experiments.md` – run batched experiments and export artifacts.

## API highlights

- `cutlass.Rectifier`: transformer implementing the critical-range binarisation.
- `cutlass.CutlassLogisticCV`: lower-level L1 logistic with cross-validation.
- `cutlass.CutlassClassifier`: full workflow composed of the rectifier,
  optional scaling, and the logistic path solver.
- `cutlass.serialization`: helpers for saving rectifier limits and fitted
  weights.

Refer to the docstrings for detailed parameter descriptions; they mirror the
research scripts so existing experiment drivers can be migrated with minimal
changes.

## Development

To build the package locally:

```bash
python -m build
```

Run the unit tests (if any) with:

```bash
python -m pytest
```

## License

MIT License.  See `LICENSE` for details.
