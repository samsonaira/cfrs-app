"""ML pipeline and recommendation engine tests."""
from django.test import TestCase
import pandas as pd

from ml.preprocessing.pipeline import clean, build_interactions
from ml.recommendation.hybrid import hybrid_score, recommend_for_buyer, DEFAULT_WEIGHTS
from ml.recommendation.preference import build_buyer_preferences


def _sample_df():
    return pd.DataFrame([
        {"transaction_id": f"T{i}", "buyer_id": "B1", "farmer_id": "F1",
         "produce_type": "Maize", "quantity": 100, "unit": "kg", "price": 45,
         "location": "Kiambu", "transaction_date": "2023-01-01", "status": "COMPLETED"}
        for i in range(5)
    ] + [
        {"transaction_id": "T99", "buyer_id": "B1", "farmer_id": "F2",
         "produce_type": "Rice", "quantity": 50, "unit": "kg", "price": 120,
         "location": "Kisumu", "transaction_date": "2023-02-01", "status": "COMPLETED"},
    ])


class PreprocessingTests(TestCase):
    def test_clean_removes_cancelled_and_invalid(self):
        df = _sample_df()
        df.loc[len(df)] = {"transaction_id": "TX", "buyer_id": "B2",
            "farmer_id": "F1", "produce_type": "Maize", "quantity": -5,
            "unit": "kg", "price": 45, "location": "Kiambu",
            "transaction_date": "2023-01-01", "status": "COMPLETED"}
        clean_df, _ = clean(df)
        self.assertTrue((clean_df["quantity"] > 0).all())

    def test_interaction_matrix_built(self):
        clean_df, _ = clean(_sample_df())
        inter = build_interactions(clean_df)
        # B1 interacted with F1 (5x) and F2 (1x)
        self.assertEqual(len(inter), 2)
        f1 = inter[inter.farmer_id == "F1"].iloc[0]
        f2 = inter[inter.farmer_id == "F2"].iloc[0]
        self.assertGreater(f1["rating"], f2["rating"])  # more freq = higher rating


class HybridScoreTests(TestCase):
    def test_weighted_formula(self):
        # cf raw 5 -> normalised 1.0; produce 1.0; location 1.0 => 1.0
        score, cf_norm = hybrid_score(5.0, 1.0, 1.0)
        self.assertAlmostEqual(score, 1.0, places=3)
        self.assertAlmostEqual(cf_norm, 1.0, places=3)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0, places=6)

    def test_produce_component_contributes(self):
        # cf neutral(1->0), only produce preference present
        score, _ = hybrid_score(1.0, 1.0, 0.0)
        self.assertAlmostEqual(score, DEFAULT_WEIGHTS["produce"], places=3)


class ColdStartTests(TestCase):
    def test_cold_start_mode_for_unknown_buyer(self):
        clean_df, _ = clean(_sample_df())
        prefs = build_buyer_preferences(clean_df)
        listings = [{"listing_id": 1, "farmer_id": "F1", "produce_type": "Maize",
                     "location": "Kiambu", "quantity": 100, "unit": "kg",
                     "asking_price": 45, "farmer_name": "F1"}]
        mode, recs = recommend_for_buyer("UNKNOWN", listings, prefs,
                                         cf_algo=None, top_n=5)
        self.assertEqual(mode, "COLD_START")
        self.assertEqual(len(recs), 1)  # still returns, no crash

    def test_hybrid_mode_for_known_buyer(self):
        clean_df, _ = clean(_sample_df())
        prefs = build_buyer_preferences(clean_df)
        listings = [{"listing_id": 1, "farmer_id": "F1", "produce_type": "Maize",
                     "location": "Kiambu", "quantity": 100, "unit": "kg",
                     "asking_price": 45, "farmer_name": "F1"}]

        class FakeAlgo:
            def predict(self, u, i):
                class P: est = 3.0
                return P()
        mode, recs = recommend_for_buyer("B1", listings, prefs,
                                         cf_algo=FakeAlgo(), top_n=5)
        self.assertEqual(mode, "HYBRID")
        # B1 strongly prefers Maize/Kiambu -> produce & location score > 0
        self.assertGreater(recs[0]["produce_score"], 0)
        self.assertGreater(recs[0]["location_score"], 0)


class ColdStartProfileFallbackTests(TestCase):
    """A buyer with no history is ranked by their stated profile preferences."""

    def setUp(self):
        from users.models import Account, BuyerProfile
        from marketplace.models import ProduceListing
        farmer = Account.objects.create_user(email_verified=True, 
            username="csf@t.ke", email="csf@t.ke", password="x", role="FARMER")
        for produce, location in [("Tea", "Mombasa"), ("Maize", "Kiambu"),
                                  ("Maize", "Nakuru"), ("Rice", "Kiambu")]:
            ProduceListing.objects.create(
                farmer=farmer, produce_name=produce, quantity=100,
                asking_price=50, location=location, status="ACTIVE")
        self.buyer = Account.objects.create_user(email_verified=True, 
            username="csb@t.ke", email="csb@t.ke", password="x", role="BUYER")
        self.profile = BuyerProfile.objects.create(
            account=self.buyer, preferred_produce="maize, Beans",
            preferred_location="kiambu")

    def test_stated_preferences_rank_cold_start_results(self):
        from recommendations.services import get_recommendations
        mode, recs = get_recommendations(self.buyer, persist=False)
        self.assertEqual(mode, "COLD_START")
        ranked = [(r["produce_type"], r["location"], r["hybrid_score"]) for r in recs]
        w = DEFAULT_WEIGHTS
        self.assertEqual(ranked[0], ("Maize", "Kiambu",
                                     round(w["produce"] + w["location"], 4)))
        self.assertEqual(ranked[1], ("Maize", "Nakuru", w["produce"]))
        self.assertEqual(ranked[2], ("Rice", "Kiambu", w["location"]))
        self.assertEqual(ranked[3], ("Tea", "Mombasa", 0.0))
        self.assertIn("Matches your preferred produce", recs[0]["reasons"])

    def test_api_returns_nonzero_matches_for_new_buyer(self):
        from django.test import Client
        c = Client(); c.force_login(self.buyer)
        data = c.get("/api/recommendations/").json()
        self.assertEqual(data["recommendation_mode"], "COLD_START")
        self.assertEqual(data["recommendations"][0]["match_percent"], 60)

    def test_no_stated_preferences_still_returns_listings(self):
        from recommendations.services import get_recommendations
        self.profile.preferred_produce = ""
        self.profile.preferred_location = ""
        self.profile.save()
        mode, recs = get_recommendations(self.buyer, persist=False)
        self.assertEqual(mode, "COLD_START")
        self.assertEqual(len(recs), 4)


class RecommendationWriteSafetyTests(TestCase):
    """Item 4: GET must not write; POST (own) writes a bounded snapshot."""

    def setUp(self):
        from users.models import Account, BuyerProfile
        from marketplace.models import ProduceListing
        farmer = Account.objects.create_user(email_verified=True, 
            username="ws_f@t.ke", email="ws_f@t.ke", password="x", role="FARMER")
        for produce, location in [("Tea", "Mombasa"), ("Maize", "Kiambu"),
                                  ("Maize", "Nakuru"), ("Rice", "Kiambu")]:
            ProduceListing.objects.create(
                farmer=farmer, produce_name=produce, quantity=100,
                asking_price=50, location=location, status="ACTIVE")
        self.buyer = Account.objects.create_user(email_verified=True, 
            username="ws_b@t.ke", email="ws_b@t.ke", password="x", role="BUYER")
        BuyerProfile.objects.create(
            account=self.buyer, preferred_produce="maize", preferred_location="kiambu")
        self.admin = Account.objects.create_user(email_verified=True, 
            username="ws_a@t.ke", email="ws_a@t.ke", password="x", role="ADMINISTRATOR")

    def _count(self):
        from marketplace.models import RecommendationRecord
        return RecommendationRecord.objects.count()

    def test_get_own_does_not_write(self):
        from django.test import Client
        c = Client(); c.force_login(self.buyer)
        resp = c.get("/api/recommendations/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["recommendations"])      # still returns data
        self.assertEqual(self._count(), 0)                   # but writes nothing

    def test_get_admin_viewing_other_buyer_does_not_write(self):
        from django.test import Client
        c = Client(); c.force_login(self.admin)
        resp = c.get(f"/api/recommendations/{self.buyer.username}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self._count(), 0)

    def test_post_own_writes_bounded_snapshot_and_replaces(self):
        from django.test import Client
        from marketplace.models import RecommendationRecord
        c = Client(); c.force_login(self.buyer)
        self.assertEqual(c.post("/api/recommendations/").status_code, 200)
        first = self._count()
        self.assertGreater(first, 0)
        self.assertLessEqual(first, 10)
        self.assertTrue(RecommendationRecord.objects.filter(buyer=self.buyer).exists())
        # posting again replaces, never appends -> count stays the same
        self.assertEqual(c.post("/api/recommendations/").status_code, 200)
        self.assertEqual(self._count(), first)

    def test_admin_post_to_other_buyer_does_not_write(self):
        from django.test import Client
        c = Client(); c.force_login(self.admin)
        resp = c.post(f"/api/recommendations/{self.buyer.username}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self._count(), 0)


class RealActivityInfluencesRecsTests(TestCase):
    """Item 2: a buyer's real completed purchases (DB, keyed by account) drive recs."""

    def setUp(self):
        from users.models import Account
        from marketplace.models import ProduceListing
        self.farmer = Account.objects.create_user(email_verified=True, 
            username="ra_f@t.ke", email="ra_f@t.ke", password="x", role="FARMER")
        self.maize = ProduceListing.objects.create(
            farmer=self.farmer, produce_name="Maize", quantity=100,
            asking_price=45, location="Kiambu", status="ACTIVE")
        ProduceListing.objects.create(
            farmer=self.farmer, produce_name="Tea", quantity=100,
            asking_price=400, location="Mombasa", status="ACTIVE")
        # A real platform buyer (email username, not in the synthetic model)
        self.buyer = Account.objects.create_user(email_verified=True, 
            username="ra_b@t.ke", email="ra_b@t.ke", password="x", role="BUYER")

    def _complete_a_maize_purchase(self):
        from marketplace.services.transaction_service import (
            create_transaction, confirm_transaction)
        tx = create_transaction(self.buyer, self.maize.pk, 10)
        confirm_transaction(self.farmer, tx.pk)

    def test_db_preferences_keyed_by_account(self):
        from recommendations.services import _db_transaction_preferences
        self.assertIsNone(_db_transaction_preferences(self.buyer))   # no history yet
        self._complete_a_maize_purchase()
        prefs = _db_transaction_preferences(self.buyer)
        self.assertEqual(prefs["produce"], {"Maize": 1.0})
        self.assertEqual(prefs["location"], {"Kiambu": 1.0})

    def test_completed_purchase_drives_recommendations(self):
        from recommendations.services import get_recommendations
        # before any purchase: not HISTORY (no real activity to use)
        mode0, _ = get_recommendations(self.buyer, persist=False)
        self.assertNotEqual(mode0, "HISTORY")
        # after a real Maize purchase, recs are driven by that history
        self._complete_a_maize_purchase()
        mode, recs = get_recommendations(self.buyer, persist=False)
        self.assertEqual(mode, "HISTORY")
        self.assertEqual(recs[0]["produce_type"], "Maize")
        self.assertGreater(recs[0]["hybrid_score"], 0)

    def test_other_buyers_history_does_not_leak(self):
        from users.models import Account
        from recommendations.services import _db_transaction_preferences
        self._complete_a_maize_purchase()
        other = Account.objects.create_user(email_verified=True, 
            username="ra_b2@t.ke", email="ra_b2@t.ke", password="x", role="BUYER")
        self.assertIsNone(_db_transaction_preferences(other))  # keyed by account
