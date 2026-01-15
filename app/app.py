from __future__ import annotations

import csv
import json
import os
import time
import zlib
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

app = Flask(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "diamonds_price_regressor")
LOGS_DIR = Path(os.getenv("LOGS_DIR", Path(__file__).resolve().parents[1] / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REQUESTS_PATH = LOGS_DIR / "requests.csv"

_split_a = int(os.getenv("AB_SPLIT_A", "70"))
_model_cache: dict[str, tuple[str, object]] = {}


def _ensure_requests_header() -> None:
    if REQUESTS_PATH.exists() and REQUESTS_PATH.stat().st_size > 0:
        return
    with REQUESTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "user_id",
                "route",
                "model_version",
                "features",
                "prediction",
                "true_price",
                "latency_ms",
            ]
        )


def _hash_bucket(user_id: int) -> int:
    return abs(zlib.crc32(str(user_id).encode("utf-8"))) % 100


def _route_for_user(user_id: int) -> str:
    return "A" if _hash_bucket(user_id) < _split_a else "B"


def _load_model(stage: str) -> tuple[object, str]:
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not versions:
        raise RuntimeError(f"No model in stage {stage}")
    version = versions[0].version

    cached = _model_cache.get(stage)
    if cached and cached[0] == version:
        return cached[1], version

    model_uri = f"models:/{MODEL_NAME}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    _model_cache[stage] = (version, model)
    return model, version


@app.post("/predict")
def predict():
    start = time.time()
    payload = request.get_json(force=True)

    if "user_id" not in payload:
        return jsonify({"error": "user_id is required"}), 400

    user_id = int(payload["user_id"])
    route = _route_for_user(user_id)
    stage = "Production" if route == "A" else "Staging"

    try:
        model, version = _load_model(stage)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    features = {
        "carat": payload.get("carat"),
        "cut": payload.get("cut"),
        "color": payload.get("color"),
        "clarity": payload.get("clarity"),
        "depth": payload.get("depth"),
        "table": payload.get("table"),
        "x": payload.get("x"),
        "y": payload.get("y"),
        "z": payload.get("z"),
    }

    df = pd.DataFrame([features])
    pred = float(model.predict(df)[0])

    latency_ms = int((time.time() - start) * 1000)
    true_price = payload.get("true_price")

    _ensure_requests_header()
    with REQUESTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                user_id,
                route,
                version,
                json.dumps(features),
                pred,
                true_price,
                latency_ms,
            ]
        )

    return jsonify(
        {
            "route": route,
            "stage": stage,
            "model_version": version,
            "prediction": pred,
            "latency_ms": latency_ms,
        }
    )


@app.post("/config")
def update_config():
    global _split_a
    payload = request.get_json(force=True)
    if "ab_split_a" not in payload:
        return jsonify({"error": "ab_split_a is required"}), 400

    new_split = int(payload["ab_split_a"])
    if new_split < 0 or new_split > 100:
        return jsonify({"error": "ab_split_a must be 0-100"}), 400

    _split_a = new_split
    return jsonify({"ab_split_a": _split_a})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
