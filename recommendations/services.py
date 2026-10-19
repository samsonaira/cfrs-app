"""
Recommendation service layer.

Bridges the Django ORM and the standalone ML engine. Loads trained model
artifacts once (module-level cache) and enforces the CRITICAL rule:
a farmer is only recommended for produce they CURRENTLY have as an
ACTIVE listing. The ML may suggest a farmer from history, but the
marketplace must verify current availability before display.
"""
import os
import joblib

from django.conf import settings
from marketplace.models import ProduceListing, RecommendationRecord

# ML engine imports (engine is Django-independent; the project root is on
# sys.path under manage.py, and ml/ is a proper package, so a plain import works)
ML_BASE = os.path.join(settings.BASE_DIR, "ml")
from ml.recommendation.hybrid import recommend_for_buyer

MODELS_DIR = os.path.join(ML_BASE, "models")

_ARTIFACTS = {"loaded": False, "ub_algo": None, "prefs": None}


def _load_artifacts():
    """Load trained models once and cache them in module memory."""
    if _ARTIFACTS["loaded"]:
        return _ARTIFACTS
    try:
        _ARTIFACTS["ub_algo"] = joblib.load(os.path.join(MODELS_DIR, "user_based_cf.pkl"))
        _ARTIFACTS["prefs"] = joblib.load(os.path.join(MODELS_DIR, "buyer_preferences.pkl"))
        _ARTIFACTS["loaded"] = True
    except FileNotFoundError:
        # Models not yet trained — cold-start-only mode
        _ARTIFACTS["ub_algo"] = None
        _ARTIFACTS["prefs"] = {}
        _ARTIFACTS["loaded"] = True
    return _ARTIFACTS


def _buyer_external_id(buyer_account):
    """
    Map a Django Buyer account to the external buyer_id used in the ML
    dataset. For demo/seeded buyers we store that id on the username;
    real platform buyers use their DB-derived key. The preference dict
    is keyed by the dataset id.
    """
    # Convention: seeded ML buyers have username == their dataset id (e.g. 'B0007')
    return buyer_account.username


def _profile_preferences(buyer_account):
    """
    Cold-start fallback: preference scores taken from what the buyer stated
    on their BuyerProfile (preferred_produce / preferred_location), in the
    same shape build_buyer_preferences() produces. A stated preference
    counts as a full match (1.0). Returns None if nothing is set.
    """
    profile = getattr(buyer_account, "buyer_profile", None)
    if profile is None:
        return None
    produce = {p.title(): 1.0 for p in profile.preferred_produce_list()}
    location = profile.preferred_location.strip().title()
    if not produce and not location:
        return None
    return {"produce": produce, "location": {location: 1.0} if location else {}}


def get_recommendations(buyer_account, top_n=10, weights=None, persist=True):
    """
    Returns (mode, recommendations[]) where each recommendation is backed
    by a CURRENT active listing. Farmers recommended by the ML model who
    have no matching active listing are excluded here.
    """
    art = _load_artifacts()
    buyer_id = _buyer_external_id(buyer_account)

    # 1. Gather CURRENT active listings — this is the availability ground truth
    active = (ProduceListing.objects
              .filter(status=ProduceListing.Status.ACTIVE)
              .select_related("farmer"))

    active_listings = []
    for lst in active:
        active_listings.append({
            "listing_id": lst.id,
            "farmer_id": lst.farmer.username,   # external id used by the model
            "farmer_pk": lst.farmer.pk,
            "farmer_name": lst.farmer.get_full_name() or lst.farmer.username,
            "produce_type": lst.produce_name.title(),
            "location": lst.location.title(),
            "quantity": float(lst.quantity),
            "unit": lst.unit,
            "asking_price": float(lst.asking_price),
        })

    if not active_listings:
        return "COLD_START", []

    # 2. Run the ML hybrid engine over ONLY currently-active listings.
    #    Because we score active listings directly, every recommendation
    #    returned is guaranteed to correspond to real current availability.
    prefs, cf_algo = art["prefs"], art["ub_algo"]
    if not prefs.get(buyer_id, {}).get("produce"):
        # No transaction history in the trained model: cold start. Fall back
        # to the buyer's stated produce/location preferences so matches are
        # still ranked. cf_algo=None keeps the engine in COLD_START mode.
        stated = _profile_preferences(buyer_account)
        if stated:
            prefs, cf_algo = {buyer_id: stated}, None
    mode, recs = recommend_for_buyer(
        buyer_id, active_listings, prefs,
        cf_algo=cf_algo, weights=weights, top_n=top_n,
    )

    # 3. Optionally persist for the analytics/audit trail
    if persist and recs:
        records = [
            RecommendationRecord(
                buyer=buyer_account,
                farmer_id=r["farmer_pk"],
                produce_type=r["produce_type"],
                location=r["location"],
                cf_score=r["cf_score"],
                produce_score=r["produce_score"],
                location_score=r["location_score"],
                hybrid_score=r["hybrid_score"],
                mode=mode,
            )
            for r in recs
        ]
        RecommendationRecord.objects.bulk_create(records)

    return mode, recs
