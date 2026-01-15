from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(np.min(expected), np.max(expected), bins + 1)

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    eps = 1e-6
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("DRIFT_THRESHOLD", "0.2")),
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(data_dir / "train.parquet")
    current = pd.read_parquet(data_dir / "current.parquet")

    numeric_cols = ["carat", "depth", "table", "x", "y", "z"]
    psi_scores = {}
    for col in numeric_cols:
        psi_scores[col] = _psi(train[col].values, current[col].values)

    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    drift = max_psi > args.threshold

    payload = {
        "drift": drift,
        "threshold": args.threshold,
        "max_psi": max_psi,
        "psi": psi_scores,
    }

    (logs_dir / "drift.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"drift={int(drift)}")


if __name__ == "__main__":
    main()
