"""
Custom authentication model for the CFRS platform.

The Account model replaces Django's default User so that role
(Farmer / Buyer / Administrator) is a first-class field and so the
schema stays portable to MySQL later without restructuring.
"""
import secrets

from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class Account(AbstractUser):
    """A single authentication identity that is exactly one role."""

    class Role(models.TextChoices):
        FARMER = "FARMER", "Farmer"
        BUYER = "BUYER", "Buyer"
        ADMINISTRATOR = "ADMINISTRATOR", "Administrator"

    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=20, blank=True)
    location = models.CharField(max_length=120, blank=True)
    role = models.CharField(max_length=15, choices=Role.choices, default=Role.BUYER)
    email_verified = models.BooleanField(
        default=False,
        help_text="True once the user has confirmed their email via a one-time code.",
    )

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return f"{self.get_full_name() or self.username} ({self.get_role_display()})"

    @property
    def is_farmer(self):
        return self.role == self.Role.FARMER

    @property
    def is_buyer(self):
        return self.role == self.Role.BUYER

    @property
    def is_administrator(self):
        return self.role == self.Role.ADMINISTRATOR or self.is_superuser


class FarmerProfile(models.Model):
    account = models.OneToOneField(
        Account, on_delete=models.CASCADE, related_name="farmer_profile"
    )
    farm_location = models.CharField(max_length=120)
    farm_size_acres = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    farming_experience_years = models.PositiveIntegerField(default=0)
    bio = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"FarmerProfile<{self.account.username}>"


class BuyerProfile(models.Model):
    class BuyerType(models.TextChoices):
        RETAILER = "RETAILER", "Retailer"
        WHOLESALER = "WHOLESALER", "Wholesaler"
        PROCESSOR = "PROCESSOR", "Processor"
        EXPORTER = "EXPORTER", "Exporter"
        INDIVIDUAL = "INDIVIDUAL", "Individual"

    account = models.OneToOneField(
        Account, on_delete=models.CASCADE, related_name="buyer_profile"
    )
    business_name = models.CharField(max_length=150, blank=True)
    buyer_type = models.CharField(
        max_length=15, choices=BuyerType.choices, default=BuyerType.RETAILER
    )
    preferred_location = models.CharField(max_length=120, blank=True)
    preferred_produce = models.CharField(max_length=250, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"BuyerProfile<{self.account.username}>"

    def preferred_produce_list(self):
        return [p.strip() for p in self.preferred_produce.split(",") if p.strip()]


class EmailVerificationCode(models.Model):
    """
    A single-use, time-limited one-time code (OTP) for email verification.

    The code itself is never stored in plaintext: only a salted hash is kept
    (the same hashing Django uses for passwords), so a database leak cannot
    reveal valid codes. Codes expire, allow a limited number of attempts, and
    are invalidated on use or when a newer code is issued.
    """

    EXPIRY_MINUTES = 10
    MAX_ATTEMPTS = 5
    CODE_LENGTH = 6

    account = models.ForeignKey(
        Account, on_delete=models.CASCADE, related_name="verification_codes"
    )
    code_hash = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    attempts = models.PositiveIntegerField(default=0)
    is_used = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"VerificationCode<{self.account.email} @ {self.created_at:%Y-%m-%d %H:%M}>"

    @classmethod
    def generate_plaintext(cls):
        """A cryptographically secure zero-padded numeric code."""
        return f"{secrets.randbelow(10 ** cls.CODE_LENGTH):0{cls.CODE_LENGTH}d}"

    def set_code(self, plaintext):
        self.code_hash = make_password(plaintext)

    def check_code(self, plaintext):
        return check_password((plaintext or "").strip(), self.code_hash)

    @property
    def is_expired(self):
        return timezone.now() >= self.expires_at

    @property
    def attempts_exhausted(self):
        return self.attempts >= self.MAX_ATTEMPTS

    @property
    def is_usable(self):
        return not self.is_used and not self.is_expired and not self.attempts_exhausted
