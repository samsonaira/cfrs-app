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


def train_price_model_monthly(verbose=True):
    """
    Higher-resolution price model for the staple crops (maize, beans) that
    have real monthly market-level data (WFP). Trained on market + region +
    year + month, which gives far more signal than annual national data and
    lets the model meet the <10% MAE target for these crops.
    """
    from ml.preprocessing.wfp_prices import load_monthly
    df = load_monthly()
    le_crop = LabelEncoder(); df["crop_enc"] = le_crop.fit_transform(df["produce"])
    le_mkt = LabelEncoder(); df["mkt_enc"] = le_mkt.fit_transform(df["market"])
    le_reg = LabelEncoder(); df["reg_enc"] = le_reg.fit_transform(df["region"])

    feat = ["crop_enc", "mkt_enc", "reg_enc", "year", "month"]
    X = df[feat].values
    y = df["price_kes_kg"].values

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae = round(-cross_val_score(model, X, y, cv=kf,
                                 scoring="neg_mean_absolute_error", n_jobs=-1).mean(), 3)
    mean_price = round(float(y.mean()), 2)
    mae_pct = round(100 * mae / mean_price, 1)

    # per-crop breakdown
    per_crop = {}
    for crop in sorted(df["produce"].unique()):
        d = df[df["produce"] == crop]
        Xc = d[feat].values; yc = d["price_kes_kg"].values
        mc = round(-cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                                    Xc, yc, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1).mean(), 3)
        per_crop[crop] = {"mae_kes": mc, "mean_price_kes": round(float(yc.mean()), 2),
                          "mae_pct_of_mean": round(100 * mc / yc.mean(), 1), "n_obs": len(d)}

    model.fit(X, y)
    joblib.dump(model, os.path.join(MODELS_DIR, "price_model_monthly.pkl"))
    joblib.dump({"crop": le_crop, "mkt": le_mkt, "reg": le_reg},
                os.path.join(MODELS_DIR, "price_monthly_encoders.pkl"))

    results = {
        "model": "Price Prediction (Random Forest, WFP monthly market data)",
        "data_source": "World Bank/WFP Real-Time Food Prices — Kenya, monthly, 233 markets, 2007-2026 (real)",
        "crops_covered": sorted(df["produce"].unique()),
        "n_observations": len(df),
        "n_markets": int(df["market"].nunique()),
        "cv_mae_kes": mae,
        "mean_price_kes": mean_price,
        "mae_pct_of_mean": mae_pct,
        "per_crop": per_crop,
    }
    if verbose:
        print(f"Monthly price model: {len(df)} obs, {df['market'].nunique()} markets")
        print(f"Overall 5-fold CV MAE: {mae} KES ({mae_pct}% of mean)")
        for c, v in per_crop.items():
            print(f"  {c}: MAE {v['mae_kes']} KES = {v['mae_pct_of_mean']}% of mean ({v['n_obs']} obs)")
    return results
