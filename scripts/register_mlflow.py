from __future__ import annotations

import os
from pathlib import Path

from mlflow.tracking import MlflowClient


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    logs_dir = base_dir / "logs"
    run_id = (logs_dir / "last_run.txt").read_text(encoding="utf-8").strip()

    model_name = os.getenv("MLFLOW_MODEL_NAME", "diamonds_price_regressor")
    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"model_name={model_name} version={mv.version} stage=Staging")


if __name__ == "__main__":
    main()
