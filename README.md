# MLOps Diamonds A/B

Учебный проект: drift, автообучение, регистрация модели и A/B роутинг.

## Как запустить

```bash
cp .env.example .env
docker compose up --build
```

## Интерфейсы

- Airflow: http://localhost:8080
- MLflow: http://localhost:5000
- MinIO console: http://localhost:9001

## MinIO (один раз)

MLflow использует бакет `mlflow` (см. `MLFLOW_ARTIFACT_ROOT`). Если бакета нет, он создаётся так:

```bash
docker run --rm --env-file .env --network mlops-diamonds-ab_default minio/mc \
  sh -c "mc alias set local http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD && mc mb local/mlflow"
```

Если сеть называется по-другому, используется имя из `docker network ls`.

## Ручной запуск DAG

В Airflow включается `drift_retrain_register`, затем запускается вручную через Play.

## Запросы в Flask

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "carat": 0.23, "cut": "Ideal", "color": "E", "clarity": "SI2", "depth": 61.5, "table": 55, "x": 3.95, "y": 3.98, "z": 2.43, "true_price": 326}'
```

Смена сплита без перезапуска:

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"ab_split_a": 80}'
```

## A/B отчёт

```bash
docker compose exec airflow-webserver python /opt/airflow/scripts/ab_analyze.py
```

Отчёт сохраняется в `logs/ab_report.md`.

## Логи и файлы

- Drift: `logs/drift.json`
- Последний run id: `logs/last_run.txt`
- Запросы A/B: `logs/requests.csv`
- A/B отчёт: `logs/ab_report.md`
