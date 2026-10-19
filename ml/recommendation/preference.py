"""
Buyer preference modelling from historical transactions.

Produces normalised produce-preference and location-preference scores
per buyer, used by the hybrid recommender and cold-start path.
"""
import pandas as pd


def build_buyer_preferences(clean_df):
    """
    Returns dict: buyer_id -> {
        'produce': {produce_type: normalised_score},
        'location': {location: normalised_score},
    }
    """
    prefs = {}
    for bid, grp in clean_df.groupby("buyer_id"):
        produce_counts = grp["produce_type"].value_counts(normalize=True).to_dict()
        location_counts = grp["location"].value_counts(normalize=True).to_dict()
        prefs[bid] = {"produce": produce_counts, "location": location_counts}
    return prefs


def produce_score(prefs, buyer_id, produce_type):
    return prefs.get(buyer_id, {}).get("produce", {}).get(produce_type, 0.0)


def location_score(prefs, buyer_id, location):
    return prefs.get(buyer_id, {}).get("location", {}).get(location, 0.0)


if __name__ == "__main__":
    # Run from the project root with: python -m ml.recommendation.preference
    from ml.preprocessing.pipeline import run
    clean_df, _, _ = run(verbose=False)
    prefs = build_buyer_preferences(clean_df)
    sample = list(prefs.keys())[0]
    print(f"Sample buyer {sample}:")
    print("  Top produce prefs:", dict(sorted(prefs[sample]['produce'].items(),
          key=lambda x: -x[1])[:3]))
    print("  Top location prefs:", dict(sorted(prefs[sample]['location'].items(),
          key=lambda x: -x[1])[:3]))
