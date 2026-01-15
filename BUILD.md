# Как запустить с нуля

Ниже короткая инструкция, как поднять проект с нуля.

## 1) Что нужно

- Docker Desktop
- Git
- Python (только если запускать `scripts/send_requests.py` локально)

На Windows должна быть включена виртуализация (Tweaker (M.I.T.) → Advanced CPU Settings → SVM Mode → Enabled).

## 2) Клонирование и .env

```bash
git clone <repo_url>
cd mlops-diamonds-ab
cp .env.example .env
```

Пароли в `.env` можно оставить по умолчанию.

## 3) Запуск сервисов

```bash
docker compose up -d --build
```

Если нужно пересобрать только app:

```bash
docker compose build app
docker compose up -d app
```

## 4) MinIO бакет (один раз)

```powershell
docker run --rm --network mlops-diamonds-ab_default `
  -e "MC_HOST_local=http://minioadmin:minioadmin@minio:9000" `
  minio/mc mb local/mlflow
```

Если сеть называется иначе, подставь имя из `docker network ls`.

## 5) Прогнать DAG

- Airflow: http://localhost:8080  
- Логин/пароль: `admin` / `admin`
- Включить `drift_retrain_register` и нажать Play.

## 6) Назначить стадии модели

MLflow: http://localhost:5000

В Model Registry для `diamonds_price_regressor`:

- одну версию перевести в **Production**
- одну оставить в **Staging**

Без Production Flask будет отдавать 503 на роут A.

## 7) Сгенерировать запросы

```bash
python scripts/send_requests.py
```

Или PowerShell-скриптом:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/send_requests.ps1
```

## 8) Сформировать A/B отчёт

```bash
docker compose exec airflow-webserver python /opt/airflow/scripts/ab_analyze.py
```

Отчёт: `logs/ab_report.md`.
