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
            password="Str0ngPass!23", role="BUYER", email_verified=True)
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
            password="Str0ngPass!23", role="BUYER", email_verified=True)
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
# --- OTP email-verification tests (ported) ---



import re
from datetime import timedelta
from unittest.mock import patch
from django.core import mail
from django.test import override_settings
from django.utils import timezone
from users.models import EmailVerificationCode
from users import otp

REG = {
    "role": "BUYER", "first_name": "Test", "email": "buyer@t.ke",
    "location": "Nakuru", "password1": "Str0ngPass!23", "password2": "Str0ngPass!23",
}


def code_from_outbox():
    """Extract the 6-digit code from the most recent email."""
    body = mail.outbox[-1].body
    return re.search(r"\b(\d{6})\b", body).group(1)


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    OTP_RESEND_COOLDOWN_SECONDS=0,
)
class EmailVerificationTests(TestCase):
    """The 12 verification scenarios required by the feature specification."""

    def setUp(self):
        self.c = Client()
        mail.outbox = []

    def _register(self, email="buyer@t.ke", role="BUYER"):
        self.c.post(reverse("register"), {**REG, "email": email, "role": role})
        return Account.objects.get(email=email)

    # 1. Registration sends a verification code.
    def test_registration_sends_code(self):
        acct = self._register()
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(EmailVerificationCode.objects.filter(account=acct).count(), 1)
        self.assertFalse(acct.email_verified)
        self.assertFalse(acct.is_active)

    # 2. The correct code verifies the account.
    def test_correct_code_verifies(self):
        acct = self._register()
        resp = self.c.post(reverse("verify_email"), {"code": code_from_outbox()})
        self.assertEqual(resp.status_code, 302)
        acct.refresh_from_db()
        self.assertTrue(acct.email_verified)
        self.assertTrue(acct.is_active)
        self.assertIn("_auth_user_id", self.c.session)  # logged in after verify

    # 3. An incorrect code is rejected.
    def test_incorrect_code_rejected(self):
        acct = self._register()
        self.c.post(reverse("verify_email"), {"code": "000000"})
        acct.refresh_from_db()
        self.assertFalse(acct.email_verified)

    # 4. An expired code is rejected.
    def test_expired_code_rejected(self):
        acct = self._register()
        real = code_from_outbox()
        code = EmailVerificationCode.objects.filter(account=acct, is_used=False).first()
        code.expires_at = timezone.now() - timedelta(minutes=1)
        code.save(update_fields=["expires_at"])
        self.c.post(reverse("verify_email"), {"code": real})
        acct.refresh_from_db()
        self.assertFalse(acct.email_verified)

    # 5. A code cannot be reused after successful verification.
    def test_code_not_reusable(self):
        acct = self._register()
        real = code_from_outbox()
        self.c.post(reverse("verify_email"), {"code": real})     # verifies, logs in
        acct.refresh_from_db()
        self.assertTrue(acct.email_verified)
        acct.email_verified = False  # pretend we try to reuse the same code
        acct.save(update_fields=["email_verified"])
        ok, _ = otp.verify(acct, real)
        self.assertFalse(ok)  # the code is already marked used

    # 6. Resending a code invalidates the previous code.
    def test_resend_invalidates_previous(self):
        acct = self._register()
        first = code_from_outbox()
        self.c.post(reverse("resend_code"))
        second = code_from_outbox()
        self.assertNotEqual(first, second)
        # old code no longer verifies
        self.assertFalse(otp.verify(acct, first)[0])
        # new code does
        self.assertTrue(otp.verify(acct, second)[0])

    # 7. Too many verification attempts are blocked.
    def test_too_many_attempts_blocked(self):
        acct = self._register()
        real = code_from_outbox()
        for _ in range(EmailVerificationCode.MAX_ATTEMPTS):
            ok, _ = otp.verify(acct, "111111")
            self.assertFalse(ok)
        # the correct code no longer works — the code was invalidated
        ok, _ = otp.verify(acct, real)
        self.assertFalse(ok)

    # 8. Unverified users cannot log in.
    def test_unverified_cannot_log_in(self):
        self._register()
        resp = self.c.post(reverse("login"),
                           {"email": "buyer@t.ke", "password": "Str0ngPass!23"})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("verify-email", resp.url)
        self.assertNotIn("_auth_user_id", self.c.session)

    # 9. Verified users can log in and reach their correct dashboard.
    def test_verified_user_logs_in_to_dashboard(self):
        self._register()
        self.c.post(reverse("verify_email"), {"code": code_from_outbox()})
        self.c.get(reverse("logout"))
        resp = self.c.post(reverse("login"),
                          {"email": "buyer@t.ke", "password": "Str0ngPass!23"},
                          follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "buyer/dashboard.html")

    # 10. Direct URL access cannot bypass verification.
    def test_direct_dashboard_access_blocked(self):
        self._register()  # not verified, not logged in
        resp = self.c.get(reverse("dashboard"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.url)

    # 11. Email delivery failures are handled without false success or leaks.
    def test_email_failure_handled(self):
        acct = Account.objects.create_user(
            username="f@t.ke", email="f@t.ke", password="Str0ngPass!23",
            role="BUYER", is_active=False)
        with patch("users.otp._send_email", side_effect=Exception("SMTP down")):
            ok, msg = otp.create_and_send(acct, force=True)
        self.assertFalse(ok)                      # no false success
        self.assertNotRegex(msg, r"\d{6}")        # no code leaked in the message
        self.assertEqual(EmailVerificationCode.objects.filter(account=acct).count(), 0)
        acct.refresh_from_db()
        self.assertFalse(acct.email_verified)

    # 12. The workflow works across the Farmer, Buyer and Administrator roles.
    def test_roles_farmer_buyer_admin(self):
        # Farmer: verify then reach farmer dashboard
        self._register(email="farmer@t.ke", role="FARMER")
        self.c.post(reverse("verify_email"), {"code": code_from_outbox()})
        r = self.c.get(reverse("dashboard"), follow=True)
        self.assertTemplateUsed(r, "farmer/dashboard.html")
        self.c.get(reverse("logout"))

        # Buyer: verify then reach buyer dashboard
        self._register(email="buyer2@t.ke", role="BUYER")
        self.c.post(reverse("verify_email"), {"code": code_from_outbox()})
        r = self.c.get(reverse("dashboard"), follow=True)
        self.assertTemplateUsed(r, "buyer/dashboard.html")
        self.c.get(reverse("logout"))

        # Administrator (superuser) is exempt from email verification
        Account.objects.create_superuser(
            username="admin@t.ke", email="admin@t.ke", password="Str0ngPass!23",
            role="ADMINISTRATOR")
        resp = self.c.post(reverse("login"),
                          {"email": "admin@t.ke", "password": "Str0ngPass!23"})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("_auth_user_id", self.c.session)


class SocialAuthTests(TestCase):
    """Google/GitHub social login: the verified-email policy and wiring."""

    def _sociallogin(self, email, verified, role_session="BUYER"):
        from allauth.socialaccount.models import SocialLogin, SocialAccount
        from allauth.account.models import EmailAddress
        user = Account(email=email, username=email, first_name="Soc")
        account = SocialAccount(provider="google", uid="uid-" + email,
                                extra_data={"email": email, "email_verified": verified})
        ea = EmailAddress(email=email, verified=verified, primary=True)
        return SocialLogin(user=user, account=account, email_addresses=[ea])

    def _request(self, role="BUYER"):
        from django.test import RequestFactory
        from importlib import import_module
        from django.conf import settings
        req = RequestFactory().get("/accounts/google/login/callback/")
        engine = import_module(settings.SESSION_ENGINE)
        req.session = engine.SessionStore()
        req.session["social_signup_role"] = role
        return req

    def test_provider_verified_flag(self):
        from users.adapters import provider_email_is_verified
        self.assertTrue(provider_email_is_verified(self._sociallogin("v@x.ke", True)))
        self.assertFalse(provider_email_is_verified(self._sociallogin("u@x.ke", False)))

    def test_social_signup_verified_creates_active_verified_account(self):
        from users.adapters import CFRSSocialAccountAdapter
        req = self._request("FARMER")
        sl = self._sociallogin("gfarm@x.ke", verified=True)
        adapter = CFRSSocialAccountAdapter()
        adapter.populate_user(req, sl, {"email": "gfarm@x.ke"})
        user = adapter.save_user(req, sl)
        user.refresh_from_db()
        self.assertTrue(user.email_verified)
        self.assertTrue(user.is_active)
        self.assertEqual(user.role, "FARMER")
        self.assertTrue(FarmerProfile.objects.filter(account=user).exists())
        self.assertEqual(len(mail.outbox), 0)  # verified: no OTP sent

    def test_social_signup_unverified_requires_otp(self):
        from users.adapters import CFRSSocialAccountAdapter
        req = self._request("BUYER")
        sl = self._sociallogin("gunv@x.ke", verified=False)
        adapter = CFRSSocialAccountAdapter()
        adapter.populate_user(req, sl, {"email": "gunv@x.ke"})
        user = adapter.save_user(req, sl)
        user.refresh_from_db()
        self.assertFalse(user.email_verified)       # untrusted email: not verified
        self.assertEqual(len(mail.outbox), 1)       # OTP sent instead
        self.assertTrue(BuyerProfile.objects.filter(account=user).exists())

    def test_social_start_sets_role_and_redirects(self):
        c = Client()
        resp = c.get(reverse("social_start", args=["github"]) + "?role=FARMER")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, "/accounts/github/login/")
        self.assertEqual(c.session.get("social_signup_role"), "FARMER")

    def test_social_buttons_render(self):
        c = Client()
        for path in ("/login/", "/register/"):
            html = c.get(path).content.decode()
            self.assertIn("/social/google/start/", html)
            self.assertIn("/social/github/start/", html)


