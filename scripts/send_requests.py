from __future__ import annotations

import json
import random
import time
import urllib.request


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def main() -> None:
    url = "http://localhost:8000/predict"
    for user_id in range(1, 21):
        payload = {
            "user_id": user_id,
            "carat": 0.3 + random.random() * 0.2,
            "cut": "Ideal",
            "color": "E",
            "clarity": "SI2",
            "depth": 61.5,
            "table": 55,
            "x": 4.3,
            "y": 4.35,
            "z": 2.7,
            "true_price": 500 + random.randint(-50, 50),
        }
        result = _post(url, payload)
        print(user_id, result)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
