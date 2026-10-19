"""
Price prediction module — trained on REAL FAO Kenya Producer Price data.

Uses Random Forest Regression to predict a crop's price (KES/kg) from
produce type and year. All metrics come from a real held-out test split.
Because the dataset is small (192 real annual observations) we report
honest cross-validated error rather than a single optimistic split.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from ml.preprocessing.fao_prices import load_clean

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_price_model(verbose=True):
    df = load_clean()
    le = LabelEncoder()
    df["produce_enc"] = le.fit_transform(df["produce"])

    X = df[["produce_enc", "year"]].values
    y = df["price_kes_kg"].values

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)

    # Honest evaluation: 5-fold CV on real data (small dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_mae = cross_val_score(model, X, y, cv=kf,
                              scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(model, X, y, cv=kf,
                               scoring="neg_root_mean_squared_error")
    mae = round(-neg_mae.mean(), 3)
    rmse = round(-neg_rmse.mean(), 3)
    mean_price = round(float(y.mean()), 2)
    mae_pct = round(100 * mae / mean_price, 1)

    # Fit final model on all real data for inference
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODELS_DIR, "price_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "price_label_encoder.pkl"))

    results = {
        "model": "Price Prediction (Random Forest, FAO Kenya data)",
        "data_source": "FAOSTAT Producer Prices — Kenya, 2010-2024 (real)",
        "n_observations": len(df),
        "produce_types": int(df["produce"].nunique()),
        "cv_mae_kes": mae,
        "cv_rmse_kes": rmse,
        "mean_price_kes": mean_price,
        "mae_pct_of_mean": mae_pct,
    }
    if verbose:
        print(f"Trained on {len(df)} real FAO observations, "
              f"{df['produce'].nunique()} produce types")
        print(f"5-fold CV MAE: {mae} KES/kg ({mae_pct}% of mean price {mean_price})")
        print(f"5-fold CV RMSE: {rmse} KES/kg")
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


def predict_price(produce, year=2025):
    model = _load_cached("price_model.pkl")
    le = _load_cached("price_label_encoder.pkl")
    if produce not in le.classes_:
        return None
    enc = le.transform([produce])[0]
    pred = model.predict([[enc, year]])[0]
    return round(float(pred), 2)


if __name__ == "__main__":
    res = train_price_model()
    print("\nSample predictions for 2025 (KES/kg):")
    for p in ["Potatoes", "Rice", "Tomatoes", "Beans", "Wheat"]:
        pred = predict_price(p, 2025)
        if pred: print(f"  {p}: KES {pred}/kg")
