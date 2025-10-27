#!/usr/bin/env python3

"""
Prepare binary CUTLASS-ready datasets from the leukemia_gene_expression.csv file.

Creates four CSVs:
  - leukemia_aml_indc.csv       (INDC=1 for AML)
  - leukemia_all_indc.csv       (INDC=1 for ALL)
  - leukemia_aml_all_indc.csv   (INDC=1 for AML or ALL)
  - leukemia_healthy_indc.csv   (INDC=1 for Healthy)
Each output drops the original Diagnosis column.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def make_variant(df: pd.DataFrame, positive_labels: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["INDC"] = out["Diagnosis"].isin(positive_labels).astype(int)
    return out.drop(columns=["Diagnosis"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CUTLASS-ready leukemia CSVs.")
    parser.add_argument(
        "--input",
        default="sample_data/leukemia_gene_expression.csv",
        help="Path to the source leukemia CSV (default: sample_data/leukemia_gene_expression.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_data",
        help="Directory to place the generated CSV files (default: sample_data)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "Diagnosis" not in df.columns:
        raise ValueError("Input CSV must contain a 'Diagnosis' column.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "leukemia_aml_indc.csv": {"AML"},
        "leukemia_all_indc.csv": {"ALL"},
        "leukemia_aml_all_indc.csv": {"AML", "ALL"},
        "leukemia_healthy_indc.csv": {"Healthy"},
    }

    for filename, positives in variants.items():
        out_df = make_variant(df, positives)
        out_path = output_dir / filename
        out_df.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
