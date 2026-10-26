"""
Training orchestrator: trains all models, evaluates them, and saves
artifacts + real evaluation results to disk. Called by the
`train_recommender` management command.
"""
import json
import os
import joblib

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_all(verbose=True):
    from ml.preprocessing.pipeline import run
    from ml.recommendation.preference import build_buyer_preferences
    from ml.recommendation.collaborative import (
        train_user_based, train_item_based, popularity_baseline, evaluate,
    )

    clean_df, interactions, stats = run(verbose=False)

    ub_algo, ub_preds, _, _ = train_user_based(interactions)
    ib_algo, ib_preds, _, _ = train_item_based(interactions)
    pop_preds = popularity_baseline(interactions)

    results = {
        "dataset": {
            "n_buyers": stats["n_buyers"],
            "n_farmers": stats["n_farmers"],
            "n_interactions": stats["n_interactions"],
            "matrix_cells": stats["matrix_cells"],
            "density": stats["density"],
            "sparsity": stats["sparsity"],
        },
        "hybrid_weights": {"cf": 0.40, "produce": 0.35, "location": 0.25},
        "models": [
            evaluate(pop_preds, "Popularity Baseline"),
            evaluate(ub_preds, "User-Based CF"),
            evaluate(ib_preds, "Item-Based CF"),
        ],
    }

    prefs = build_buyer_preferences(clean_df)

    # --- Additional ML modules (real / legitimately-synthetic data) ---
    from ml.recommendation.price_prediction import train_price_model, train_price_model_monthly
    from ml.recommendation.demand_forecast import train_demand_model
    from ml.recommendation.fraud_detection import train_fraud_model
    import pandas as _pd

    price_res = train_price_model(verbose=False)
    price_monthly_res = train_price_model_monthly(verbose=False)
    demand_res = train_demand_model(verbose=False)
    _txdf = clean_df.rename(columns={})  # already has needed cols
    fraud_res = train_fraud_model(_txdf, verbose=False)

    results["price_prediction"] = price_res
    results["price_prediction_monthly"] = price_monthly_res
    results["demand_forecasting"] = demand_res
    results["fraud_detection"] = fraud_res

    # Persist artifacts
    joblib.dump(ub_algo, os.path.join(MODELS_DIR, "user_based_cf.pkl"))
    joblib.dump(ib_algo, os.path.join(MODELS_DIR, "item_based_cf.pkl"))
    joblib.dump(prefs, os.path.join(MODELS_DIR, "buyer_preferences.pkl"))
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    if verbose:
        print("Artifacts saved to", MODELS_DIR)
        print(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    train_all()
