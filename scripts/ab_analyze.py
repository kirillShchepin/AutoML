from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def _format_float(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    logs_dir = base_dir / "logs"
    requests_path = logs_dir / "requests.csv"
    report_path = logs_dir / "ab_report.md"

    if not requests_path.exists() or requests_path.stat().st_size == 0:
        report_path.write_text("No requests found.\n", encoding="utf-8")
        return

    df = pd.read_csv(requests_path)
    if df.empty:
        report_path.write_text("No requests found.\n", encoding="utf-8")
        return

    if "true_price" in df.columns:
        df["true_price"] = pd.to_numeric(df["true_price"], errors="coerce")
    else:
        df["true_price"] = np.nan

    if "prediction" in df.columns:
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    else:
        df["prediction"] = np.nan

    if "latency_ms" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    else:
        df["latency_ms"] = np.nan

    metrics = []
    errors_by_route = {}

    for route in ["A", "B"]:
        subset = df[df["route"] == route]
        with_true = subset.dropna(subset=["true_price", "prediction"])
        rmse = None
        mae = None
        if not with_true.empty:
            errors = with_true["prediction"] - with_true["true_price"]
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mae = float(np.mean(np.abs(errors)))
            errors_by_route[route] = np.abs(errors).values
        else:
            errors_by_route[route] = np.array([])

        metrics.append(
            {
                "route": route,
                "requests": int(len(subset)),
                "latency_ms": float(subset["latency_ms"].mean())
                if not subset.empty
                else float("nan"),
                "rmse": rmse,
                "mae": mae,
                "n_true": int(len(with_true)),
            }
        )

    p_value = None
    if len(errors_by_route["A"]) > 1 and len(errors_by_route["B"]) > 1:
        _, p_value = ttest_ind(
            errors_by_route["A"], errors_by_route["B"], equal_var=False
        )

    recommendation = "Insufficient data"
    mae_a = next((m["mae"] for m in metrics if m["route"] == "A"), None)
    mae_b = next((m["mae"] for m in metrics if m["route"] == "B"), None)
    if p_value is not None and mae_a is not None and mae_b is not None:
        if p_value < 0.05 and mae_b < mae_a:
            recommendation = "Promote B to Production"
        else:
            recommendation = "Keep A in Production"

    lines = ["# A/B Report", "", "## Metrics", "", "| Route | Requests | N True | Mean latency (ms) | RMSE | MAE |", "| --- | --- | --- | --- | --- | --- |"]
    for m in metrics:
        lines.append(
            "| {route} | {requests} | {n_true} | {latency} | {rmse} | {mae} |".format(
                route=m["route"],
                requests=m["requests"],
                n_true=m["n_true"],
                latency=_format_float(m["latency_ms"]),
                rmse=_format_float(m["rmse"]),
                mae=_format_float(m["mae"]),
            )
        )

    lines.extend(
        [
            "",
            "## Significance",
            "",
            f"p-value: {_format_float(p_value)}",
            "",
            f"Recommendation: {recommendation}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
