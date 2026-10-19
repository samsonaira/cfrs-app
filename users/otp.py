"""
Email one-time-code (OTP) service for CFRS account verification.

Responsibilities:
- generate a cryptographically secure 6-digit code (via the model),
- store only its hash, with an expiry and attempt limit,
- send it by email using Django's configured backend,
- enforce resend rate limits,
- verify a submitted code and activate the account on success.

Security notes:
- Plaintext codes are never stored or logged; only a salted hash is kept.
- A code is invalidated on successful use, on expiry, when attempts run out,
  and when a newer code is issued for the same account.
- Rate limits cap how often codes can be requested.
"""
from datetime import timedelta

from django.conf import settings
from django.core.mail import send_mail
from django.utils import timezone

from .models import EmailVerificationCode


def _resend_cooldown_seconds():
    return getattr(settings, "OTP_RESEND_COOLDOWN_SECONDS", 60)


def _max_codes_per_hour():
    return getattr(settings, "OTP_MAX_CODES_PER_HOUR", 5)


def _invalidate_active(account):
    """Mark every not-yet-used code for this account as used (superseded)."""
    EmailVerificationCode.objects.filter(account=account, is_used=False).update(is_used=True)


def can_send(account):
    """Rate-limit check. Returns (allowed: bool, message: str)."""
    now = timezone.now()
    recent = EmailVerificationCode.objects.filter(
        account=account, created_at__gte=now - timedelta(hours=1)
    )
    if recent.count() >= _max_codes_per_hour():
        return False, "Too many codes requested recently. Please try again later."
    last = recent.order_by("-created_at").first()
    if last:
        elapsed = (now - last.created_at).total_seconds()
        cooldown = _resend_cooldown_seconds()
        if elapsed < cooldown:
            wait = int(cooldown - elapsed)
            return False, f"Please wait {wait}s before requesting a new code."
    return True, ""


def create_and_send(account, force=False):
    """
    Issue a fresh code and email it. Supersedes any previous code.
    `force=True` skips the resend cooldown (used for the first send at
    registration). Returns (ok: bool, message: str).
    """
    if not force:
        allowed, msg = can_send(account)
        if not allowed:
            return False, msg

    _invalidate_active(account)

    plaintext = EmailVerificationCode.generate_plaintext()
    code = EmailVerificationCode(
        account=account,
        expires_at=timezone.now() + timedelta(minutes=EmailVerificationCode.EXPIRY_MINUTES),
    )
    code.set_code(plaintext)
    code.save()

    try:
        _send_email(account, plaintext)
    except Exception:
        # Delivery failed: do not falsely claim success, and do not leave an
        # undelivered code lying around. The plaintext is never logged.
        code.delete()
        return False, "We couldn't send the verification email just now. Please try again shortly."

    return True, "A 6-digit verification code has been sent to your email."


def _send_email(account, plaintext):
    subject = "Your CFRS verification code"
    minutes = EmailVerificationCode.EXPIRY_MINUTES
    body = (
        f"Hello {account.first_name or ''},\n\n"
        f"Your CFRS email verification code is: {plaintext}\n\n"
        f"It expires in {minutes} minutes. Enter it on the verification page to "
        f"activate your account.\n\n"
        f"If you did not create a CFRS account, you can ignore this email.\n"
    )
    send_mail(
        subject,
        body,
        getattr(settings, "DEFAULT_FROM_EMAIL", "no-reply@cfrs.local"),
        [account.email],
        fail_silently=False,
    )


def verify(account, submitted):
    """
    Check a submitted code against the account's most recent unused code.
    On success, marks the code used and activates the account.
    Returns (ok: bool, message: str).
    """
    code = (
        EmailVerificationCode.objects.filter(account=account, is_used=False)
        .order_by("-created_at")
        .first()
    )
    if code is None:
        return False, "No active code. Please request a new one."
    if code.is_expired:
        return False, "This code has expired. Please request a new one."
    if code.attempts_exhausted:
        code.is_used = True
        code.save(update_fields=["is_used"])
        return False, "Too many incorrect attempts. Please request a new code."

    if code.check_code(submitted):
        code.is_used = True
        code.save(update_fields=["is_used"])
        account.email_verified = True
        account.is_active = True
        account.save(update_fields=["email_verified", "is_active"])
        return True, "Your email has been verified."

    # Wrong code: count the attempt, invalidate if the limit is now reached.
    code.attempts += 1
    code.save(update_fields=["attempts"])
    remaining = code.MAX_ATTEMPTS - code.attempts
    if remaining <= 0:
        code.is_used = True
        code.save(update_fields=["is_used"])
        return False, "Incorrect code. Too many attempts — please request a new code."
    return False, f"Incorrect code. {remaining} attempt(s) remaining."
