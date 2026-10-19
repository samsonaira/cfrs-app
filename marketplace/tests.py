"""Marketplace, transaction and authorization tests."""
from decimal import Decimal
from django.conf import settings
from django.test import TestCase, Client
from django.core.exceptions import ValidationError, PermissionDenied
from users.models import Account
from marketplace.models import ProduceListing, Transaction
from marketplace.services.transaction_service import (
    create_transaction, confirm_transaction, cancel_transaction, TransactionError,
)


class ListingModelTests(TestCase):
    def setUp(self):
        self.farmer = Account.objects.create_user(
            username="f@t.ke", email="f@t.ke", password="x", role="FARMER")

    def test_valid_listing(self):
        l = ProduceListing(farmer=self.farmer, produce_name="Maize",
                           quantity=100, asking_price=45, location="Kiambu")
        l.full_clean()  # should not raise

    def test_zero_quantity_rejected(self):
        l = ProduceListing(farmer=self.farmer, produce_name="Maize",
                           quantity=0, asking_price=45, location="Kiambu")
        with self.assertRaises(ValidationError):
            l.full_clean()

    def test_negative_price_rejected(self):
        l = ProduceListing(farmer=self.farmer, produce_name="Maize",
                           quantity=100, asking_price=-5, location="Kiambu")
        with self.assertRaises(ValidationError):
            l.full_clean()


class CreateListingViewTests(TestCase):
    def setUp(self):
        self.farmer = Account.objects.create_user(
            username="lf@t.ke", email="lf@t.ke", password="x", role="FARMER")
        self.c = Client(); self.c.force_login(self.farmer)

    def _post(self, quantity, price):
        return self.c.post("/listings/new/", {
            "produce_name": "Rice", "quantity": quantity,
            "asking_price": price, "location": "Nakuru", "unit": "kg"})

    def test_two_decimal_price_and_quantity_accepted(self):
        resp = self._post("0.1", "45.10")
        self.assertRedirects(resp, "/dashboard/")
        listing = ProduceListing.objects.get(farmer=self.farmer)
        self.assertEqual(listing.quantity, Decimal("0.10"))
        self.assertEqual(listing.asking_price, Decimal("45.10"))

    def test_invalid_numbers_rejected_without_error(self):
        for qty, price in [("abc", "10"), ("nan", "10"), ("inf", "10"),
                           ("10", ""), ("-5", "10"), ("10", "45.123")]:
            resp = self._post(qty, price)
            self.assertRedirects(resp, "/listings/new/",
                                 fetch_redirect_response=False)
        self.assertEqual(ProduceListing.objects.count(), 0)


class TransactionServiceTests(TestCase):
    def setUp(self):
        self.farmer = Account.objects.create_user(
            username="f2@t.ke", email="f2@t.ke", password="x", role="FARMER")
        self.buyer = Account.objects.create_user(
            username="b2@t.ke", email="b2@t.ke", password="x", role="BUYER")
        self.listing = ProduceListing.objects.create(
            farmer=self.farmer, produce_name="Rice", quantity=100,
            asking_price=120, location="Kisumu", status="ACTIVE")

    def test_total_amount_calculation(self):
        tx = create_transaction(self.buyer, self.listing.pk, 10)
        self.assertEqual(tx.total_amount, Decimal("1200.00"))
        self.assertEqual(tx.status, "PENDING")

    def test_over_quantity_rejected(self):
        with self.assertRaises(TransactionError):
            create_transaction(self.buyer, self.listing.pk, 999)

    def test_zero_quantity_rejected(self):
        with self.assertRaises(TransactionError):
            create_transaction(self.buyer, self.listing.pk, 0)

    def test_non_finite_quantity_rejected(self):
        for bad in ("nan", "NaN", "Infinity", "-inf", "abc", ""):
            with self.assertRaises(TransactionError):
                create_transaction(self.buyer, self.listing.pk, bad)

    def test_too_many_decimal_places_rejected(self):
        with self.assertRaises(TransactionError):
            create_transaction(self.buyer, self.listing.pk, "0.001")

    def test_two_decimal_quantity_accepted(self):
        tx = create_transaction(self.buyer, self.listing.pk, "10.50")
        self.assertEqual(tx.quantity, Decimal("10.50"))

    def test_buy_view_bad_quantity_shows_message_not_500(self):
        c = Client(); c.force_login(self.buyer)
        for bad in ("nan", "0.001"):
            resp = c.post(f"/marketplace/{self.listing.pk}/buy/",
                          {"quantity": bad}, follow=True)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(list(resp.context["messages"])), 1)
        self.assertEqual(Transaction.objects.count(), 0)

    def test_confirm_decrements_stock(self):
        tx = create_transaction(self.buyer, self.listing.pk, 40)
        confirm_transaction(self.farmer, tx.pk)
        self.listing.refresh_from_db(); tx.refresh_from_db()
        self.assertEqual(self.listing.quantity, Decimal("60.00"))
        self.assertEqual(tx.status, "COMPLETED")

    def test_wrong_farmer_cannot_confirm(self):
        other = Account.objects.create_user(
            username="f3@t.ke", email="f3@t.ke", password="x", role="FARMER")
        tx = create_transaction(self.buyer, self.listing.pk, 10)
        with self.assertRaises(PermissionDenied):
            confirm_transaction(other, tx.pk)

    def test_inactive_listing_rejected(self):
        self.listing.status = "INACTIVE"; self.listing.save()
        with self.assertRaises(TransactionError):
            create_transaction(self.buyer, self.listing.pk, 10)

    def test_sold_out_marks_listing_sold(self):
        tx = create_transaction(self.buyer, self.listing.pk, 100)
        confirm_transaction(self.farmer, tx.pk)
        self.listing.refresh_from_db()
        self.assertEqual(self.listing.status, "SOLD")

    def test_cancel_transaction(self):
        tx = create_transaction(self.buyer, self.listing.pk, 10)
        cancel_transaction(self.buyer, tx.pk)
        tx.refresh_from_db()
        self.assertEqual(tx.status, "CANCELLED")


class AuthorizationTests(TestCase):
    def setUp(self):
        self.farmer = Account.objects.create_user(
            username="af@t.ke", email="af@t.ke", password="x", role="FARMER")
        self.buyer = Account.objects.create_user(
            username="ab@t.ke", email="ab@t.ke", password="x", role="BUYER")

    def test_farmer_cannot_get_recommendations(self):
        c = Client(); c.force_login(self.farmer)
        resp = c.get("/api/recommendations/")
        self.assertEqual(resp.status_code, 403)

    def test_buyer_cannot_access_admin_dashboard(self):
        c = Client(); c.force_login(self.buyer)
        resp = c.get("/admin-dashboard/")
        self.assertIn(resp.status_code, (302, 403))  # redirected/denied

    def test_unauthenticated_recommendations_blocked(self):
        c = Client()
        resp = c.get("/api/recommendations/")
        self.assertEqual(resp.status_code, 403)

    def test_public_pages_load(self):
        c = Client()
        for url in ["/", "/login/", "/register/", "/marketplace/"]:
            self.assertEqual(c.get(url).status_code, 200)


class AdminDashboardEscapingTests(TestCase):
    def test_listing_name_cannot_break_out_of_script_block(self):
        payload = "</script><script>alert(2)</script>"
        ProduceListing.objects.create(
            farmer=Account.objects.create_user(
                username="xf@t.ke", email="xf@t.ke", password="x", role="FARMER"),
            produce_name=payload, quantity=7, asking_price=9,
            location=payload, status="ACTIVE")
        admin = Account.objects.create_user(
            username="ad@t.ke", email="ad@t.ke", password="x",
            role="ADMINISTRATOR")
        c = Client(); c.force_login(admin)
        resp = c.get("/admin-dashboard/")
        self.assertEqual(resp.status_code, 200)
        html = resp.content.decode()
        self.assertNotIn(payload, html)
        self.assertNotIn("<script>alert(2)", html)
        # the name is still delivered to the chart, safely encoded
        self.assertIn("alert(2)", html)


class DjangoAdminTests(TestCase):
    def test_models_are_manageable_in_django_admin(self):
        farmer = Account.objects.create_user(
            username="daf@t.ke", email="daf@t.ke", password="x", role="FARMER")
        buyer = Account.objects.create_user(
            username="dab@t.ke", email="dab@t.ke", password="x", role="BUYER")
        listing = ProduceListing.objects.create(
            farmer=farmer, produce_name="Rice", quantity=100,
            asking_price=120, location="Kisumu", status="ACTIVE")
        create_transaction(buyer, listing.pk, 10)
        su = Account.objects.create_superuser(
            username="su@t.ke", email="su@t.ke", password="x")
        c = Client(); c.force_login(su)
        for url in ["/admin/", "/admin/users/account/", "/admin/users/account/add/",
                    f"/admin/users/account/{farmer.pk}/change/",
                    "/admin/users/farmerprofile/", "/admin/users/buyerprofile/",
                    "/admin/marketplace/producelisting/",
                    f"/admin/marketplace/producelisting/{listing.pk}/change/",
                    "/admin/marketplace/transaction/",
                    "/admin/marketplace/recommendationrecord/"]:
            self.assertEqual(c.get(url).status_code, 200, url)


class EvaluationVerdictTests(TestCase):
    """Item 1: the analytics verdict must follow the numbers, not overclaim."""

    def _results(self, base, user, item):
        keys = ["rmse", "mae", "precision_at_5", "recall_at_5", "f1_at_5"]
        mk = lambda name, vals: {"model": name, **dict(zip(keys, vals))}
        return {
            "dataset": {"sparsity": 0.94841},
            "models": [mk("Popularity Baseline", base),
                       mk("User-Based CF", user),
                       mk("Item-Based CF", item)],
        }

    def test_recall_and_error_not_claimed_when_baseline_wins(self):
        from marketplace.views import _evaluation_verdict
        # Real deterministic numbers: CF wins P@5/F1@5, baseline wins R@5/RMSE/MAE.
        r = self._results(
            base=[0.8259, 0.6518, 0.3389, 0.6699, 0.4501],
            user=[0.8616, 0.6713, 0.3585, 0.6666, 0.4663],
            item=[0.8593, 0.6852, 0.3499, 0.6525, 0.4555])
        v = _evaluation_verdict(r)
        self.assertEqual(v["cf_better"], ["Precision@5", "F1@5"])
        self.assertIn("Recall@5", v["baseline_better"])
        self.assertIn("RMSE", v["baseline_better"])
        self.assertIn("MAE", v["baseline_better"])
        # Must NOT repeat the old overclaim that CF beats baseline on Recall@5.
        self.assertNotIn("on Precision@5, Recall@5 and F1@5", v["sentence"])
        self.assertIn("baseline records the best", v["sentence"])

    def test_verdict_tracks_the_numbers_if_cf_wins_everything(self):
        from marketplace.views import _evaluation_verdict
        # Hypothetical run where CF beats baseline on all ranking metrics.
        r = self._results(
            base=[0.90, 0.70, 0.30, 0.60, 0.40],
            user=[0.80, 0.60, 0.35, 0.65, 0.45],
            item=[0.81, 0.61, 0.34, 0.64, 0.44])
        v = _evaluation_verdict(r)
        self.assertEqual(v["cf_better"], ["Precision@5", "Recall@5", "F1@5", "RMSE", "MAE"])
        self.assertEqual(v["baseline_better"], [])

    def test_analytics_page_has_no_hardcoded_claim(self):
        tpl = (settings.BASE_DIR / "templates" / "dashboard_admin" / "ml_analytics.html").read_text()
        self.assertNotIn("outperform the", tpl)
        self.assertIn("eval_summary.sentence", tpl)


class DemoCredentialsTests(TestCase):
    """Item 6: the admin superuser must not share the published demo password."""

    def test_admin_superuser_not_given_shared_demo_password(self):
        from marketplace.management.commands.seed_demo_data import Command, DEMO_PASSWORD
        cmd = Command()
        cmd._demo_accounts()
        admin = Account.objects.get(username="demo_admin")
        self.assertTrue(admin.is_superuser)
        self.assertFalse(admin.check_password(DEMO_PASSWORD))
        self.assertTrue(admin.check_password(cmd.admin_password))
        self.assertGreaterEqual(len(cmd.admin_password), 12)
        # low-privilege demo logins still use the shared demo password
        self.assertTrue(Account.objects.get(username="demo_buyer").check_password(DEMO_PASSWORD))

    def test_admin_password_from_env_when_set(self):
        import os
        os.environ["CFRS_ADMIN_PASSWORD"] = "env-set-admin-pw-123"
        try:
            from marketplace.management.commands.seed_demo_data import Command
            cmd = Command()
            cmd._demo_accounts()
            self.assertTrue(Account.objects.get(username="demo_admin")
                            .check_password("env-set-admin-pw-123"))
        finally:
            del os.environ["CFRS_ADMIN_PASSWORD"]


class AdminDashboardResilienceTests(TestCase):
    """Item 9a: dashboard must not 500 when fraud_model.pkl is untrained."""

    def test_dashboard_200_when_fraud_model_untrained(self):
        from unittest.mock import patch
        from django.urls import reverse
        farmer = Account.objects.create_user(
            username="f9@t.ke", email="f9@t.ke", password="x", role="FARMER")
        buyer = Account.objects.create_user(
            username="b9@t.ke", email="b9@t.ke", password="x", role="BUYER")
        listing = ProduceListing.objects.create(
            farmer=farmer, produce_name="Maize", quantity=100,
            asking_price=45, location="Kiambu", status="ACTIVE")
        create_transaction(buyer, listing.pk, 5)
        admin = Account.objects.create_user(
            username="a9@t.ke", email="a9@t.ke", password="x", role="ADMINISTRATOR")
        c = Client(); c.force_login(admin)
        with patch("ml.recommendation.fraud_detection.score_transaction",
                   side_effect=FileNotFoundError):
            resp = c.get(reverse("admin_dashboard"))
        self.assertEqual(resp.status_code, 200)


class RequestMessageRoundingTests(TestCase):
    """Item 9b: the 'Request sent' total is shown rounded to 2 dp."""

    def test_total_rounded_to_two_dp(self):
        import re
        from django.urls import reverse
        farmer = Account.objects.create_user(
            username="f10@t.ke", email="f10@t.ke", password="x", role="FARMER")
        buyer = Account.objects.create_user(
            username="b10@t.ke", email="b10@t.ke", password="x", role="BUYER")
        listing = ProduceListing.objects.create(
            farmer=farmer, produce_name="Maize", quantity=Decimal("100"),
            asking_price=Decimal("45.15"), location="Kiambu", status="ACTIVE")
        c = Client(); c.force_login(buyer)
        resp = c.post(reverse("start_transaction", args=[listing.pk]),
                      {"quantity": "0.10"}, follow=True)
        msgs = " ".join(str(m) for m in list(resp.context["messages"]))
        self.assertIn("Total KSh", msgs)
        self.assertNotRegex(msgs, r"KSh \d+\.\d{3,}")  # never 3+ decimal places


class PendingReservationTests(TestCase):
    """Item 5: outstanding pending requests reserve stock at request time."""

    def setUp(self):
        farmer = Account.objects.create_user(
            username="pr_f@t.ke", email="pr_f@t.ke", password="x", role="FARMER")
        self.b1 = Account.objects.create_user(
            username="pr_b1@t.ke", email="pr_b1@t.ke", password="x", role="BUYER")
        self.b2 = Account.objects.create_user(
            username="pr_b2@t.ke", email="pr_b2@t.ke", password="x", role="BUYER")
        self.listing = ProduceListing.objects.create(
            farmer=farmer, produce_name="Maize", quantity=Decimal("100"),
            asking_price=50, location="Kiambu", status="ACTIVE")

    def test_pending_requests_cannot_oversubscribe_stock(self):
        create_transaction(self.b1, self.listing.pk, Decimal("60"))   # 40 left
        with self.assertRaises(TransactionError):
            create_transaction(self.b2, self.listing.pk, Decimal("50"))  # too much now
        tx = create_transaction(self.b2, self.listing.pk, Decimal("40"))  # exactly fits
        self.assertEqual(tx.status, Transaction.Status.PENDING)
        with self.assertRaises(TransactionError):
            create_transaction(self.b1, self.listing.pk, Decimal("1"))   # nothing left

    def test_cancelling_a_pending_request_frees_the_reservation(self):
        t1 = create_transaction(self.b1, self.listing.pk, Decimal("100"))
        with self.assertRaises(TransactionError):
            create_transaction(self.b2, self.listing.pk, Decimal("10"))
        cancel_transaction(self.b1, t1.pk)
        tx = create_transaction(self.b2, self.listing.pk, Decimal("10"))
        self.assertEqual(tx.status, Transaction.Status.PENDING)
