"""
Quickstart example for the CUTLASS package.

The script mirrors the usage pattern described in README.md.  Run with:

    python examples/quickstart.py
"""

from __future__ import annotations

import pandas as pd

from cutlass import CutlassClassifier


def main() -> None:
    df = pd.DataFrame(
        {
            "feat_a": [0.1, 0.3, 0.7, 0.9, 0.2, 0.8],
            "feat_b": [10, 13, 8, 5, 11, 4],
            "INDC": [0, 0, 1, 1, 0, 1],
        }
    )
    X = df.drop(columns=["INDC"])
    y = df["INDC"]

    clf = CutlassClassifier(rectify=True, logic_polish=True, logic_scale=8.0)
    clf.fit(X, y)

    prob = clf.predict_proba(X)
    preds = clf.predict(X)
    print("probabilities:\\n", prob)
    print("predictions:", preds.tolist())
    if clf.rectify:
        print("rectifier limits:", clf.limits_)


if __name__ == "__main__":
    main()

