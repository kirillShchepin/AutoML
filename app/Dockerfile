FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 8000
CMD ["python", "/app/app.py"]
