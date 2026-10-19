"""
Collaborative filtering engines (user-based & item-based) using Surprise,
plus a popularity baseline, and Precision/Recall/F1@K evaluation.

All metrics returned are computed by real evaluation code on a held-out
test set. Nothing here is hard-coded.
"""
import os
from collections import defaultdict

import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split


def _to_surprise(interactions):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(
        interactions[["buyer_id", "farmer_id", "rating"]], reader
    )


def train_user_based(interactions, test_size=0.2, seed=42):
    data = _to_surprise(interactions)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=seed)
    algo = KNNWithMeans(k=30, sim_options={"name": "cosine", "user_based": True},
                        verbose=False)
    algo.fit(trainset)
    preds = algo.test(testset)
    return algo, preds, trainset, testset


def train_item_based(interactions, test_size=0.2, seed=42):
    data = _to_surprise(interactions)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=seed)
    algo = KNNWithMeans(k=30, sim_options={"name": "cosine", "user_based": False},
                        verbose=False)
    algo.fit(trainset)
    preds = algo.test(testset)
    return algo, preds, trainset, testset


def precision_recall_at_k(predictions, k=5, threshold=2.0):
    """Standard Surprise Precision@K / Recall@K implementation."""
    user_est_true = defaultdict(list)
    for uid, _iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_e, true_r) in ratings)
        n_rec_k = sum((est >= threshold) for (est, _t) in ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold)
                              for (est, true_r) in ratings[:k] if est >= threshold)
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    p = sum(precisions.values()) / len(precisions)
    r = sum(recalls.values()) / len(recalls)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0
    return round(p, 4), round(r, 4), round(f1, 4)


def evaluate(preds, label, k=5):
    rmse = round(accuracy.rmse(preds, verbose=False), 4)
    mae = round(accuracy.mae(preds, verbose=False), 4)
    p, r, f1 = precision_recall_at_k(preds, k=k)
    return {
        "model": label,
        "rmse": rmse, "mae": mae,
        f"precision_at_{k}": p, f"recall_at_{k}": r, f"f1_at_{k}": f1,
    }


def popularity_baseline(interactions, test_size=0.2, seed=42):
    """Non-personalised baseline: predict each farmer's global mean rating."""
    data = _to_surprise(interactions)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=seed)
    global_mean = trainset.global_mean
    item_means = {}
    for iid in trainset.all_items():
        ratings = [r for (_u, r) in trainset.ir[iid]]
        item_means[trainset.to_raw_iid(iid)] = sum(ratings) / len(ratings)
    preds = []
    for (uid, iid, true_r) in testset:
        est = item_means.get(iid, global_mean)
        preds.append((uid, iid, true_r, est, None))
    return preds


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from ml.preprocessing.pipeline import run

    _, interactions, stats = run(verbose=False)
    print(f"Interactions: {len(interactions)} | sparsity: {stats['sparsity']}\n")

    print("Training user-based CF...")
    _, ub_preds, _, _ = train_user_based(interactions)
    ub = evaluate(ub_preds, "User-Based CF")

    print("Training item-based CF...")
    _, ib_preds, _, _ = train_item_based(interactions)
    ib = evaluate(ib_preds, "Item-Based CF")

    print("Popularity baseline...")
    pop_preds = popularity_baseline(interactions)
    pop = evaluate(pop_preds, "Popularity Baseline")

    print("\n=== REAL EVALUATION RESULTS ===")
    for m in (pop, ub, ib):
        print(m)
