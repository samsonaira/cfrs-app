"""
Fraud / anomaly detection module using Isolation Forest.

Operates on transaction-level features to flag statistically anomalous
transactions for REVIEW — never to accuse. Terminology throughout is
"potential anomaly / requires review", per responsible-use guidance.

Features are engineered from real transaction structure:
  - transaction amount (quantity * price)
  - quantity
  - unit price
  - price deviation from that produce's typical price
  - buyer transaction frequency
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def engineer_features(df):
    """df needs: buyer_id, produce_type, quantity, price."""
    df = df.copy()
    df["amount"] = df["quantity"] * df["price"]
    # typical price per produce -> deviation
    typical = df.groupby("produce_type")["price"].transform("median")
    df["price_deviation"] = (df["price"] - typical).abs() / typical.replace(0, 1)
    # buyer frequency
    freq = df.groupby("buyer_id")["quantity"].transform("count")
    df["buyer_freq"] = freq
    return df[["amount", "quantity", "price", "price_deviation", "buyer_freq"]]


def train_fraud_model(df, contamination=0.03, verbose=True):
    X = engineer_features(df).fillna(0).values
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42)
    model.fit(X)
    joblib.dump(model, os.path.join(MODELS_DIR, "fraud_model.pkl"))

    scores = model.decision_function(X)   # higher = more normal
    preds = model.predict(X)              # -1 = anomaly, 1 = normal
    n_flagged = int((preds == -1).sum())

    results = {
        "model": "Fraud Detection (Isolation Forest)",
        "n_transactions": len(df),
        "n_flagged_for_review": n_flagged,
        "flagged_pct": round(100 * n_flagged / len(df), 2),
        "contamination": contamination,
        "features": ["amount", "quantity", "price", "price_deviation", "buyer_freq"],
    }
    if verbose:
        print(f"Trained on {len(df)} transactions")
        print(f"Flagged {n_flagged} ({results['flagged_pct']}%) as potential "
              f"anomalies requiring review")
    return results


_MODEL_CACHE = {}


def _load_cached(filename):
    """
    Load a saved model once and reuse it. Unpickling on every call made each
    request re-read a multi-megabyte file. The file's modification time is
    part of the cache check, so retraining is picked up without a restart.
    Raises FileNotFoundError if the model has not been trained yet.
    """
    path = os.path.join(MODELS_DIR, filename)
    mtime = os.path.getmtime(path)
    cached = _MODEL_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        cached = (mtime, joblib.load(path))
        _MODEL_CACHE[path] = cached
    return cached[1]


def score_transaction(amount, quantity, price, price_deviation, buyer_freq):
    """Returns (is_anomaly, anomaly_score). is_anomaly True => requires review."""
    model = _load_cached("fraud_model.pkl")
    X = [[amount, quantity, price, price_deviation, buyer_freq]]
    pred = model.predict(X)[0]
    score = float(model.decision_function(X)[0])
    return (pred == -1), round(score, 4)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "data", "transactions.csv"))
    df = df[df["status"] == "COMPLETED"].copy()
    res = train_fraud_model(df)

    # demonstrate: a clearly abnormal transaction should be flagged
    print("\nScoring test transactions:")
    normal = score_transaction(amount=9000, quantity=200, price=45,
                               price_deviation=0.05, buyer_freq=20)
    print(f"  Normal-looking tx:     anomaly={normal[0]}  score={normal[1]}")
    suspicious = score_transaction(amount=5_000_000, quantity=100000, price=0.5,
                                   price_deviation=0.95, buyer_freq=1)
    print(f"  Suspicious tx (huge qty, near-zero price): anomaly={suspicious[0]}  score={suspicious[1]}")
