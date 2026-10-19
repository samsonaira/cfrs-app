"""
Custom authentication model for the CFRS platform.

The Account model replaces Django's default User so that role
(Farmer / Buyer / Administrator) is a first-class field and so the
schema stays portable to MySQL later without restructuring.
"""
from django.contrib.auth.models import AbstractUser
from django.db import models


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
