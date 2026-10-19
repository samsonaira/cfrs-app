"""Marketplace domain models: ProduceListing and Transaction."""
from decimal import Decimal
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class ProduceListing(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE", "Active"
        SOLD = "SOLD", "Sold"
        EXPIRED = "EXPIRED", "Expired"
        INACTIVE = "INACTIVE", "Inactive"

    farmer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="listings", limit_choices_to={"role": "FARMER"},
    )
    produce_name = models.CharField(max_length=80)
    quantity = models.DecimalField(max_digits=12, decimal_places=2)
    unit = models.CharField(max_length=20, default="kg")
    asking_price = models.DecimalField(max_digits=12, decimal_places=2)
    location = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.ACTIVE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["produce_name"]),
            models.Index(fields=["location"]),
        ]

    def __str__(self):
        return f"{self.produce_name} — {self.quantity}{self.unit} @ {self.location}"

    def clean(self):
        if self.quantity is None or self.quantity <= 0:
            raise ValidationError({"quantity": "Quantity must be greater than zero."})
        if self.asking_price is None or self.asking_price <= 0:
            raise ValidationError({"asking_price": "Asking price must be greater than zero."})
        if not self.produce_name.strip():
            raise ValidationError({"produce_name": "Produce name is required."})
        if not self.location.strip():
            raise ValidationError({"location": "Location is required."})

    @property
    def is_active(self):
        return self.status == self.Status.ACTIVE


class Transaction(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        COMPLETED = "COMPLETED", "Completed"
        CANCELLED = "CANCELLED", "Cancelled"

    buyer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="purchases", limit_choices_to={"role": "BUYER"},
    )
    farmer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="sales", limit_choices_to={"role": "FARMER"},
    )
    listing = models.ForeignKey(
        ProduceListing, on_delete=models.SET_NULL, null=True,
        related_name="transactions",
    )
    quantity = models.DecimalField(max_digits=12, decimal_places=2)
    agreed_price = models.DecimalField(max_digits=12, decimal_places=2)
    total_amount = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    transaction_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-transaction_date"]

    def __str__(self):
        return f"Tx#{self.pk}: {self.buyer.username} -> {self.farmer.username} [{self.status}]"

    def clean(self):
        if self.quantity is None or self.quantity <= 0:
            raise ValidationError({"quantity": "Quantity must be greater than zero."})
        if self.agreed_price is None or self.agreed_price <= 0:
            raise ValidationError({"agreed_price": "Agreed price must be greater than zero."})
        if self.listing and self.status == self.Status.PENDING:
            if not self.listing.is_active:
                raise ValidationError("Cannot transact against an inactive listing.")
            if self.quantity > self.listing.quantity:
                raise ValidationError(
                    {"quantity": "Requested quantity exceeds available quantity."}
                )

    def save(self, *args, **kwargs):
        self.total_amount = (self.quantity or Decimal(0)) * (self.agreed_price or Decimal(0))
        super().save(*args, **kwargs)


class RecommendationRecord(models.Model):
    """Persisted recommendation output for audit and the ML analytics page."""
    class Mode(models.TextChoices):
        COLLABORATIVE = "COLLABORATIVE", "Collaborative"
        HYBRID = "HYBRID", "Hybrid"
        COLD_START = "COLD_START", "Cold start"

    buyer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="recommendation_records",
    )
    farmer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="recommended_in",
    )
    produce_type = models.CharField(max_length=80)
    location = models.CharField(max_length=120, blank=True)
    cf_score = models.FloatField(default=0)
    produce_score = models.FloatField(default=0)
    location_score = models.FloatField(default=0)
    hybrid_score = models.FloatField(default=0)
    mode = models.CharField(max_length=15, choices=Mode.choices, default=Mode.HYBRID)
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-generated_at"]
