"""
Preprocessing pipeline for CFRS recommendation training.

Turns the raw transaction log into a buyer-farmer interaction matrix
where the implicit 'rating' is derived from repeat purchase frequency —
a standard implicit-feedback approach for collaborative filtering.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_raw(path=None):
    path = path or os.path.join(DATA_DIR, "transactions.csv")
    return pd.read_csv(path)


def clean(df):
    n0 = len(df)
    df = df.drop_duplicates(subset=["transaction_id"])
    # required fields
    df = df.dropna(subset=["buyer_id", "farmer_id", "produce_type",
                           "quantity", "price", "location"])
    # valid numeric ranges
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    # normalise categoricals
    df["produce_type"] = df["produce_type"].str.strip().str.title()
    df["location"] = df["location"].str.strip().str.title()
    # only completed transactions represent a real buyer-farmer interaction
    df = df[df["status"] == "COMPLETED"].copy()
    return df, n0


def build_interactions(df):
    """
    Buyer-farmer interaction strength = number of completed transactions,
    mapped to an explicit 1-5 rating scale for the Surprise library.
    """
    inter = (df.groupby(["buyer_id", "farmer_id"])
               .size().reset_index(name="freq"))
    # map frequency -> 1..5 rating (log-scaled so heavy repeats don't dominate)
    inter["rating"] = 1 + 4 * (np.log1p(inter["freq"]) /
                               np.log1p(inter["freq"].max()))
    inter["rating"] = inter["rating"].round(3)
    return inter


def run(path=None, verbose=True):
    df = load_raw(path)
    clean_df, n0 = clean(df)
    interactions = build_interactions(clean_df)
    stats = {
        "raw_rows": n0,
        "clean_rows": len(clean_df),
        "n_buyers": clean_df["buyer_id"].nunique(),
        "n_farmers": clean_df["farmer_id"].nunique(),
        "n_interactions": len(interactions),
        "matrix_cells": clean_df["buyer_id"].nunique() * clean_df["farmer_id"].nunique(),
    }
    stats["density"] = round(stats["n_interactions"] / stats["matrix_cells"], 5)
    stats["sparsity"] = round(1 - stats["density"], 5)
    if verbose:
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return clean_df, interactions, stats


if __name__ == "__main__":
    run()
