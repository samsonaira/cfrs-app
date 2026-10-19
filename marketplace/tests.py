"""Marketplace, transaction and authorization tests."""
from decimal import Decimal
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
