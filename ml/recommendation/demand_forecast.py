"""
Demand forecasting module — trained on REAL FAO Kenya production data.

Uses Gradient Boosting Regression to forecast national annual production
(tonnes) per crop, a legitimate proxy for market supply/demand trend.
Honestly scoped: national + annual, not per-county/per-month (FAO has no
such granularity). All metrics from real cross-validation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                    "fao_kenya_production.csv")

FAO_TO_MARKET = {
    "Beans, dry": "Beans", "Mangoes, guavas and mangosteens": "Mangoes",
    "Cabbages": "Cabbage", "Avocados": "Avocado", "Oranges": "Oranges",
    "Sweet potatoes": "Sweet potatoes", "Potatoes": "Potatoes", "Rice": "Rice",
    "Tomatoes": "Tomatoes", "Wheat": "Wheat", "Bananas": "Bananas", "Sorghum": "Sorghum",
}


def load_clean():
    df = pd.read_csv(DATA)
    df = df[df["Element"] == "Production"][["Item", "Year", "Value"]].dropna()
    df = df[df["Value"] > 0]
    df["produce"] = df["Item"].map(FAO_TO_MARKET).fillna(df["Item"])
    df["year"] = df["Year"].astype(int)
    df["production_tonnes"] = df["Value"].astype(float)
    # lag feature: previous year's production per crop (real time-series signal)
    df = df.sort_values(["produce", "year"])
    df["prev_year_production"] = df.groupby("produce")["production_tonnes"].shift(1)
    return df[["produce", "year", "production_tonnes", "prev_year_production"]].dropna()


def train_demand_model(verbose=True):
    df = load_clean()
    le = LabelEncoder()
    df["produce_enc"] = le.fit_transform(df["produce"])
    X = df[["produce_enc", "year", "prev_year_production"]].values
    y = df["production_tonnes"].values

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae = round(-cross_val_score(model, X, y, cv=kf,
                scoring="neg_mean_absolute_error").mean(), 1)
    rmse = round(-cross_val_score(model, X, y, cv=kf,
                 scoring="neg_root_mean_squared_error").mean(), 1)
    mean_prod = round(float(y.mean()), 1)

    model.fit(X, y)
    joblib.dump(model, os.path.join(MODELS_DIR, "demand_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "demand_label_encoder.pkl"))

    results = {
        "model": "Demand Forecasting (Gradient Boosting, FAO Kenya production)",
        "data_source": "FAOSTAT Production — Kenya, 2010-2024 (real)",
        "n_observations": len(df),
        "produce_types": int(df["produce"].nunique()),
        "cv_mae_tonnes": mae,
        "cv_rmse_tonnes": rmse,
        "mean_production_tonnes": mean_prod,
        "mae_pct_of_mean": round(100 * mae / mean_prod, 1),
        "scope": "National annual production trend (not per-county/month)",
    }
    if verbose:
        print(f"Trained on {len(df)} real FAO production observations, "
              f"{df['produce'].nunique()} crops")
        print(f"5-fold CV MAE: {mae:,.0f} tonnes ({results['mae_pct_of_mean']}% of mean)")
    return results


def forecast_demand(produce, year, prev_production):
    model = joblib.load(os.path.join(MODELS_DIR, "demand_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "demand_label_encoder.pkl"))
    if produce not in le.classes_:
        return None
    enc = le.transform([produce])[0]
    pred = model.predict([[enc, year, prev_production]])[0]
    return round(float(pred), 0)


if __name__ == "__main__":
    res = train_demand_model()
    df = load_clean()
    print("\nSample: latest real production per crop (tonnes):")
    latest = df.sort_values("year").groupby("produce").tail(1)
    for _, r in latest.head(6).iterrows():
        print(f"  {r['produce']}: {r['production_tonnes']:,.0f} t ({r['year']})")
