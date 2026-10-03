"""Replay patients from the cleaned dataset against the API to generate monitoring traffic.

Usage: python scripts/load_test.py http://127.0.0.1:8000 --requests 400 --invalid-rate 0.05
About 5 % of requests are deliberately invalid so the 422/validation panels have data too.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal"]


def payloads(n: int, invalid_rate: float, rng: random.Random):
    df = pd.read_csv(ROOT / "data" / "processed" / "heart_clean.csv")
    for _ in range(n):
        row = df.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        p = {k: (None if pd.isna(row[k]) else (float(row[k]) if k == "oldpeak" else int(row[k])))
             for k in FEATURES}
        if rng.random() < invalid_rate:
            p[rng.choice(["age", "chol", "thal"])] = rng.choice([999, -5, 4])
        yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--invalid-rate", type=float, default=0.05)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()
    rng = random.Random(7)
    codes: Counter = Counter()
    labels: Counter = Counter()
    for p in payloads(args.requests, args.invalid_rate, rng):
        req = urllib.request.Request(f"{args.base_url}/predict", data=json.dumps(p).encode(),
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                codes[r.status] += 1
                labels[json.load(r)["risk_band"]] += 1
        except urllib.error.HTTPError as e:
            codes[e.code] += 1
        time.sleep(args.delay)
    print("status codes:", dict(codes))
    print("risk bands  :", dict(labels))


if __name__ == "__main__":
    main()
