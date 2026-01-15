from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    rng = np.random.default_rng(42)
    df = sns.load_dataset("diamonds")
    train = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    current = train.copy()

    numeric_cols = ["carat", "depth", "table", "x", "y", "z"]
    for col in numeric_cols:
        shift = rng.normal(loc=0.0, scale=0.5)
        scale = rng.uniform(0.9, 1.1)
        current[col] = (current[col] * scale) + shift

    if "cut" in current.columns:
        mask = rng.random(len(current)) < 0.3
        current.loc[mask, "cut"] = "Premium"

    for col in numeric_cols:
        current[col] = current[col].clip(lower=0.01)

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(data_dir / "train.parquet", index=False)
    current.to_parquet(data_dir / "current.parquet", index=False)


if __name__ == "__main__":
    main()
