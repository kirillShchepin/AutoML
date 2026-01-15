from __future__ import annotations

from pathlib import Path
import time

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from pycaret.regression import (
    compare_models,
    finalize_model,
    get_config,
    predict_model,
    setup,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _filter_params(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
    return clean


class _PyfuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(data_dir / "train.parquet")
    if len(data) > 500:
        data = data.sample(n=500, random_state=42).reset_index(drop=True)

    start_time = time.time()
    print("stage=setup start")
    setup(
        data=data,
        target="price",
        session_id=42,
        log_experiment=True,
        experiment_name="diamonds_automl",
        fold=2,
        n_jobs=-1,
        verbose=True,
    )
    print(f"stage=setup done seconds={int(time.time() - start_time)}")

    print("stage=compare_models start")
    best = compare_models(
        sort="RMSE",
        n_select=1,
        turbo=True,
        verbose=True,
        include=["lr", "ridge", "lasso"],
    )
    print(f"stage=compare_models done seconds={int(time.time() - start_time)}")
    print("stage=finalize_model start")
    final = finalize_model(best)
    print(f"stage=finalize_model done seconds={int(time.time() - start_time)}")

    print("stage=predict_model start")
    preds = predict_model(best)
    print(f"stage=predict_model done seconds={int(time.time() - start_time)}")
    y_true = get_config("y_test")
    if "Label" in preds.columns:
        y_pred = preds["Label"]
    elif "prediction_label" in preds.columns:
        y_pred = preds["prediction_label"]
    else:
        y_pred = preds.iloc[:, -1]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if mlflow.active_run() is None:
        mlflow.set_experiment("diamonds_automl")
        mlflow.start_run()

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    params = getattr(best, "get_params", lambda: {})()
    mlflow.log_params(_filter_params(params))

    try:
        mlflow_sklearn.log_model(final, "model")
    except Exception:
        mlflow.pyfunc.log_model("model", python_model=_PyfuncWrapper(final))

    run = mlflow.active_run()
    if run is not None:
        (logs_dir / "last_run.txt").write_text(run.info.run_id, encoding="utf-8")

    mlflow.end_run()


if __name__ == "__main__":
    main()
