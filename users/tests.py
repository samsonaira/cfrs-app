"""Authentication and authorization tests."""
from django.test import TestCase, Client
from django.urls import reverse
from users.models import Account, FarmerProfile, BuyerProfile


class AuthTests(TestCase):
    def test_buyer_registration_creates_account_and_profile(self):
        c = Client()
        resp = c.post(reverse("register"), {
            "role": "BUYER", "first_name": "Test", "email": "buyer@t.ke",
            "location": "Nakuru", "password1": "Str0ngPass!23",
            "password2": "Str0ngPass!23",
        })
        self.assertEqual(resp.status_code, 302)
        acct = Account.objects.get(email="buyer@t.ke")
        self.assertEqual(acct.role, "BUYER")
        self.assertTrue(BuyerProfile.objects.filter(account=acct).exists())

    def test_farmer_registration_creates_profile(self):
        c = Client()
        c.post(reverse("register"), {
            "role": "FARMER", "first_name": "Farm", "email": "farm@t.ke",
            "location": "Kiambu", "password1": "Str0ngPass!23",
            "password2": "Str0ngPass!23",
        })
        acct = Account.objects.get(email="farm@t.ke")
        self.assertEqual(acct.role, "FARMER")
        self.assertTrue(FarmerProfile.objects.filter(account=acct).exists())

    def test_login_and_logout(self):
        Account.objects.create_user(username="u@t.ke", email="u@t.ke",
            password="Str0ngPass!23", role="BUYER")
        c = Client()
        ok = c.post(reverse("login"),
                    {"email": "u@t.ke", "password": "Str0ngPass!23"})
        self.assertEqual(ok.status_code, 302)
        # logout is POST-only: a GET is refused and leaves the session alone
        self.assertEqual(c.get(reverse("logout")).status_code, 405)
        self.assertEqual(c.get(reverse("dashboard")).status_code, 200)
        c.post(reverse("logout"))
        # after logout, dashboard redirects to login
        resp = c.get(reverse("dashboard"))
        self.assertEqual(resp.status_code, 302)

    def test_sign_out_control_is_a_post_form(self):
        user = Account.objects.create_user(username="s@t.ke", email="s@t.ke",
            password="Str0ngPass!23", role="BUYER")
        c = Client(); c.force_login(user)
        html = c.get(reverse("dashboard")).content.decode()
        self.assertIn('<form method="post" action="/logout/"', html)
        self.assertNotIn('href="/logout/"', html)

    def test_invalid_login_rejected(self):
        c = Client()
        resp = c.post(reverse("login"),
                      {"email": "no@t.ke", "password": "wrong"})
        self.assertEqual(resp.status_code, 200)  # re-renders with error
        self.assertFalse("_auth_user_id" in c.session)

    def test_duplicate_email_rejected(self):
        Account.objects.create_user(username="dup@t.ke", email="dup@t.ke",
            password="x", role="BUYER")
        c = Client()
        c.post(reverse("register"), {
            "role": "BUYER", "first_name": "D", "email": "dup@t.ke",
            "password1": "Str0ngPass!23", "password2": "Str0ngPass!23"})
        self.assertEqual(Account.objects.filter(email="dup@t.ke").count(), 1)


class RolePropertyTests(TestCase):
    def test_role_helpers(self):
        f = Account(role="FARMER"); b = Account(role="BUYER")
        a = Account(role="ADMINISTRATOR")
        self.assertTrue(f.is_farmer and not f.is_buyer)
        self.assertTrue(b.is_buyer and not b.is_farmer)
        self.assertTrue(a.is_administrator)
