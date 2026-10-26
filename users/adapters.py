"""
Social-authentication adapter for CFRS (Google and GitHub via django-allauth).

Policy: an email returned by a social provider is treated as verified ONLY when
the provider itself reports it as verified. When it is verified, the account is
activated and marked verified (no OTP needed). When it is NOT verified, the
account falls back to the same one-time-code (OTP) flow used for email/password
signups. This prevents an untrusted provider email from bypassing verification.

The adapter also maps social signups onto the custom Account model: it sets the
username to the email, assigns the role chosen on the registration page (default
Buyer), and creates the matching Farmer/Buyer profile.
"""
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

from . import otp
from .models import Account, FarmerProfile, BuyerProfile


def provider_email_is_verified(sociallogin):
    """True only if the provider marks the signing-in email as verified."""
    email = (getattr(sociallogin.user, "email", "") or "").lower()
    if not email:
        return False
    # allauth populates EmailAddress objects (with a .verified flag) from the
    # provider — Google's OpenID response and GitHub's verified email list.
    for ea in getattr(sociallogin, "email_addresses", []):
        if ea.email and ea.email.lower() == email:
            return bool(ea.verified)
    # Fallback: some providers expose an explicit claim in extra_data.
    extra = getattr(sociallogin.account, "extra_data", {}) or {}
    claim = extra.get("email_verified")
    if isinstance(claim, str):
        return claim.lower() == "true"
    return bool(claim)


class CFRSSocialAccountAdapter(DefaultSocialAccountAdapter):

    def pre_social_login(self, request, sociallogin):
        """Link a social login to an existing verified account with the same email."""
        if sociallogin.is_existing:
            return
        email = (getattr(sociallogin.user, "email", "") or "").lower()
        if not email:
            return
        existing = Account.objects.filter(email__iexact=email).first()
        if existing and existing.email_verified:
            sociallogin.connect(request, existing)

    def populate_user(self, request, sociallogin, data):
        user = super().populate_user(request, sociallogin, data)
        email = (data.get("email") or getattr(user, "email", "") or "").strip()
        if email and not user.username:
            user.username = email  # Account uses email as the username
        role = request.session.get("social_signup_role", "BUYER")
        user.role = role if role in ("FARMER", "BUYER") else "BUYER"
        return user

    def save_user(self, request, sociallogin, form=None):
        user = super().save_user(request, sociallogin, form)

        verified = provider_email_is_verified(sociallogin)
        user.email_verified = verified
        user.is_active = True  # allauth logs them in; the middleware gates access
        if not user.username:
            user.username = user.email
        user.save(update_fields=["email_verified", "is_active", "username", "role"])

        # Create the role-specific profile.
        if user.role == "FARMER":
            FarmerProfile.objects.get_or_create(
                account=user, defaults={"farm_location": user.location or "Unspecified"}
            )
        else:
            BuyerProfile.objects.get_or_create(
                account=user, defaults={"preferred_location": user.location or ""}
            )

        # Unverified provider email -> require the OTP step, like email signups.
        if not verified:
            request.session["pending_verification_account_id"] = user.id
            otp.create_and_send(user, force=True)

        return user
