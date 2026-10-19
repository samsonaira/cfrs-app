"""
Transaction service — encapsulates all transaction state changes with
validation, so views stay thin and rules live in one place.
"""
from decimal import Decimal
from django.core.exceptions import ValidationError, PermissionDenied
from django.db import transaction as db_tx
from django.db.models import Sum

from marketplace.models import ProduceListing, Transaction


class TransactionError(Exception):
    """Raised for business-rule violations; carries a user-safe message."""


@db_tx.atomic
def create_transaction(buyer, listing_id, quantity):
    """Buyer initiates a PENDING transaction against an active listing."""
    try:
        quantity = Decimal(str(quantity))
        # NaN / Infinity parse as Decimals but are not usable quantities
        if not quantity.is_finite():
            raise ValueError
        rounded = quantity.quantize(Decimal("0.01"))
    except Exception:
        raise TransactionError("Enter a valid quantity.")
    if quantity <= 0:
        raise TransactionError("Quantity must be greater than zero.")
    if rounded != quantity:
        raise TransactionError("Quantity can have at most 2 decimal places.")
    quantity = rounded

    listing = ProduceListing.objects.select_for_update().filter(pk=listing_id).first()
    if not listing:
        raise TransactionError("That listing no longer exists.")
    if not listing.is_active:
        raise TransactionError("That listing is no longer available.")

    # Reserve stock at request time: pending requests already outstanding
    # against this listing count against its quantity, so several buyers can't
    # each reserve more than is actually available. (select_for_update above
    # serialises concurrent requests on this listing row.)
    reserved = (Transaction.objects
                .filter(listing=listing, status=Transaction.Status.PENDING)
                .aggregate(total=Sum("quantity"))["total"] or Decimal("0"))
    available = listing.quantity - reserved
    if available <= 0:
        raise TransactionError(
            "No stock is currently available — all of it has pending requests.")
    if quantity > available:
        raise TransactionError(
            f"Only {available} {listing.unit} available "
            f"({reserved} already has pending requests).")

    tx = Transaction(
        buyer=buyer, farmer=listing.farmer, listing=listing,
        quantity=quantity, agreed_price=listing.asking_price,
        status=Transaction.Status.PENDING,
    )
    try:
        tx.full_clean()
    except ValidationError as e:
        # model-level validation failures are user errors, not server errors
        raise TransactionError("; ".join(e.messages))
    tx.save()
    return tx


@db_tx.atomic
def confirm_transaction(farmer, tx_id):
    """Farmer confirms a pending transaction; decrements listing stock."""
    tx = Transaction.objects.select_for_update().filter(pk=tx_id).first()
    if not tx:
        raise TransactionError("Transaction not found.")
    if tx.farmer_id != farmer.id:
        raise PermissionDenied("You can only confirm your own transactions.")
    if tx.status != Transaction.Status.PENDING:
        raise TransactionError("Only pending transactions can be confirmed.")

    listing = tx.listing
    if listing:
        if tx.quantity > listing.quantity:
            raise TransactionError("Requested quantity now exceeds stock.")
        listing.quantity -= tx.quantity
        if listing.quantity == 0:
            listing.status = ProduceListing.Status.SOLD
        listing.save()

    tx.status = Transaction.Status.COMPLETED
    tx.save()
    return tx


@db_tx.atomic
def cancel_transaction(actor, tx_id):
    """Either party may cancel a pending transaction."""
    tx = Transaction.objects.select_for_update().filter(pk=tx_id).first()
    if not tx:
        raise TransactionError("Transaction not found.")
    if actor.id not in (tx.buyer_id, tx.farmer_id):
        raise PermissionDenied("You are not part of this transaction.")
    if tx.status != Transaction.Status.PENDING:
        raise TransactionError("Only pending transactions can be cancelled.")
    tx.status = Transaction.Status.CANCELLED
    tx.save()
    return tx
