"""
Hybrid recommender combining CF + produce preference + location preference.

Hybrid Score = w_cf*CF + w_produce*Produce + w_location*Location
Default weights: CF=0.40, Produce=0.35, Location=0.25 (configurable).

Also provides the cold-start path for buyers with no transaction history.
"""

DEFAULT_WEIGHTS = {"cf": 0.40, "produce": 0.35, "location": 0.25}


def _normalise_cf(cf_raw, scale=(1, 5)):
    """Map a Surprise rating estimate (1-5) into 0-1 so it is comparable
    to the preference scores which are already 0-1."""
    lo, hi = scale
    return max(0.0, min(1.0, (cf_raw - lo) / (hi - lo)))


def hybrid_score(cf_raw, produce_pref, location_pref, weights=None):
    w = weights or DEFAULT_WEIGHTS
    cf_norm = _normalise_cf(cf_raw)
    score = (w["cf"] * cf_norm
             + w["produce"] * produce_pref
             + w["location"] * location_pref)
    return round(score, 4), round(cf_norm, 4)


def build_reason(produce_pref, location_pref, cf_norm, has_history):
    reasons = []
    if produce_pref > 0.05:
        reasons.append("Matches your preferred produce")
    if location_pref > 0.05:
        reasons.append("Matches your preferred location")
    if has_history and cf_norm > 0.3:
        reasons.append("Similar buyers have purchased from this farmer")
    if not reasons:
        reasons.append("Recommended from available active listings")
    return reasons


def recommend_for_buyer(buyer_id, active_listings, prefs, cf_algo=None,
                        weights=None, top_n=10):
    """
    active_listings: list of dicts with keys
        farmer_id, produce_type, location, (+ display fields)
    prefs: output of build_buyer_preferences
    cf_algo: trained Surprise algo or None (cold start)

    Returns (mode, ranked_recommendations)
    """
    has_history = buyer_id in prefs and bool(prefs[buyer_id]["produce"])
    mode = "HYBRID" if (cf_algo and has_history) else "COLD_START"

    scored = []
    for lst in active_listings:
        fid = lst["farmer_id"]
        p_pref = prefs.get(buyer_id, {}).get("produce", {}).get(lst["produce_type"], 0.0)
        l_pref = prefs.get(buyer_id, {}).get("location", {}).get(lst["location"], 0.0)

        if mode == "HYBRID":
            try:
                cf_raw = cf_algo.predict(buyer_id, fid).est
            except Exception:
                cf_raw = 1.0
        else:
            cf_raw = 1.0  # neutral; cold start leans on preferences

        hs, cf_norm = hybrid_score(cf_raw, p_pref, l_pref, weights)
        reasons = build_reason(p_pref, l_pref, cf_norm, has_history)
        scored.append({
            **lst,
            "cf_score": round(cf_raw, 4),
            "produce_score": round(p_pref, 4),
            "location_score": round(l_pref, 4),
            "hybrid_score": hs,
            "match_percent": round(hs * 100),
            "reasons": reasons,
        })

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return mode, scored[:top_n]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from ml.preprocessing.pipeline import run
    from ml.recommendation.preference import build_buyer_preferences
    from ml.recommendation.collaborative import train_user_based

    clean_df, interactions, _ = run(verbose=False)
    prefs = build_buyer_preferences(clean_df)
    algo, _, _, _ = train_user_based(interactions)

    # simulate active listings from real farmers in the data
    farmers = clean_df[["farmer_id", "produce_type", "location"]].drop_duplicates()
    listings = farmers.head(200).to_dict("records")

    bid = clean_df["buyer_id"].iloc[0]
    mode, recs = recommend_for_buyer(bid, listings, prefs, algo, top_n=5)
    print(f"Buyer {bid} | mode: {mode}\n")
    for r in recs:
        print(f"  {r['farmer_id']} | {r['produce_type']} @ {r['location']} "
              f"| hybrid={r['hybrid_score']} ({r['match_percent']}%) "
              f"| cf={r['cf_score']} prod={r['produce_score']} loc={r['location_score']}")
        print(f"     reasons: {', '.join(r['reasons'])}")

    # Cold start test — unknown buyer
    print("\n--- COLD START (new buyer 'B9999') ---")
    mode2, recs2 = recommend_for_buyer("B9999", listings, prefs, algo, top_n=3)
    print(f"mode: {mode2}")
    for r in recs2[:3]:
        print(f"  {r['farmer_id']} | {r['produce_type']} @ {r['location']} | {r['match_percent']}%")
