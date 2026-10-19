import json, os
from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.shortcuts import render, redirect, get_object_or_404

from users.models import Account
from .models import ProduceListing, Transaction, RecommendationRecord


def marketplace(request):
    qs = ProduceListing.objects.filter(status="ACTIVE").select_related("farmer")
    q = request.GET.get("q", "").strip()
    produce = request.GET.get("produce", "").strip()
    location = request.GET.get("location", "").strip()
    sort = request.GET.get("sort", "newest")

    if q:
        qs = qs.filter(Q(produce_name__icontains=q) |
                       Q(location__icontains=q) |
                       Q(farmer__first_name__icontains=q))
    if produce:
        qs = qs.filter(produce_name__iexact=produce)
    if location:
        qs = qs.filter(location__iexact=location)

    qs = {"newest": qs.order_by("-created_at"),
          "price_low": qs.order_by("asking_price"),
          "price_high": qs.order_by("-asking_price")}.get(sort, qs)

    page = Paginator(qs, 12).get_page(request.GET.get("page"))
    produce_types = (ProduceListing.objects.filter(status="ACTIVE")
                     .values_list("produce_name", flat=True).distinct().order_by("produce_name"))
    locations = (ProduceListing.objects.filter(status="ACTIVE")
                 .values_list("location", flat=True).distinct().order_by("location"))
    return render(request, "marketplace/browse.html", {
        "page": page, "q": q, "produce": produce, "location": location, "sort": sort,
        "produce_types": produce_types, "locations": locations,
    })


@login_required
def listing_detail(request, pk):
    listing = get_object_or_404(ProduceListing, pk=pk)
    return render(request, "marketplace/detail.html", {"listing": listing})


def _is_admin(u):
    return u.is_authenticated and u.is_administrator


@user_passes_test(_is_admin)
def admin_dashboard(request):
    stats = {
        "farmers": Account.objects.filter(role="FARMER").count(),
        "buyers": Account.objects.filter(role="BUYER").count(),
        "active_listings": ProduceListing.objects.filter(status="ACTIVE").count(),
        "total_tx": Transaction.objects.count(),
        "pending_tx": Transaction.objects.filter(status="PENDING").count(),
        "completed_tx": Transaction.objects.filter(status="COMPLETED").count(),
        "cancelled_tx": Transaction.objects.filter(status="CANCELLED").count(),
        "rec_requests": RecommendationRecord.objects.values("buyer").distinct().count(),
    }
    by_produce = list(ProduceListing.objects.filter(status="ACTIVE")
                      .values("produce_name").annotate(n=Count("id")).order_by("-n")[:8])
    by_location = list(ProduceListing.objects.filter(status="ACTIVE")
                       .values("location").annotate(n=Count("id")).order_by("-n")[:8])

    # Live fraud review — score real transactions with the Isolation Forest model
    flagged = _flag_transactions()

    return render(request, "dashboard_admin/dashboard.html", {
        "flagged": flagged,
        "stats": stats,
        # rendered with |json_script in the template (escapes user-entered names)
        "produce_labels": [r["produce_name"] for r in by_produce],
        "produce_values": [r["n"] for r in by_produce],
        "location_labels": [r["location"] for r in by_location],
        "location_values": [r["n"] for r in by_location],
    })


def _evaluation_verdict(results):
    """
    Derive an honest comparison of the CF models against the popularity
    baseline straight from the evaluation numbers, so the analytics page
    states what the data shows instead of a hard-coded claim.

    Returns a dict (or None) with the metrics CF wins / loses and a
    plain-language sentence built from the actual figures.
    """
    try:
        models = {m["model"]: m for m in results["models"]}
        base = models["Popularity Baseline"]
        cfs = [models["User-Based CF"], models["Item-Based CF"]]
    except (KeyError, TypeError):
        return None

    # (label, json key, higher_is_better)
    ranking = [("Precision@5", "precision_at_5", True),
               ("Recall@5", "recall_at_5", True),
               ("F1@5", "f1_at_5", True)]
    error = [("RMSE", "rmse", False), ("MAE", "mae", False)]

    def cf_beats_all(key, higher):
        return all((c[key] > base[key]) if higher else (c[key] < base[key]) for c in cfs)

    cf_better, baseline_better = [], []
    for label, key, higher in ranking + error:
        (cf_better if cf_beats_all(key, higher) else baseline_better).append(label)

    def join(items):
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    parts = []
    if cf_better:
        parts.append("Both collaborative-filtering models outperform the "
                     f"non-personalised popularity baseline on {join(cf_better)}")
        if baseline_better:
            parts.append(f", while the baseline records the best {join(baseline_better)}")
        parts.append(".")
    else:
        parts.append("At this data density the popularity baseline matches or "
                     f"beats both collaborative-filtering models on {join(baseline_better)}; "
                     "the CF models show no consistent gain.")

    sentence = "".join(parts)

    gain = round(max(c["precision_at_5"] for c in cfs) - base["precision_at_5"], 4)
    if "Precision@5" in cf_better:
        sentence += (f" The best Precision@5 gain over the baseline is +{gain} (absolute)")
        sparsity = results.get("dataset", {}).get("sparsity")
        if sparsity is not None:
            sentence += (f", a modest margin consistent with the {round(sparsity * 100, 1)}% "
                         "sparsity of the interaction matrix")
        sentence += "."

    return {"cf_better": cf_better, "baseline_better": baseline_better,
            "precision_gain": gain, "sentence": sentence}


@user_passes_test(_is_admin)
def ml_analytics(request):
    path = os.path.join(settings.BASE_DIR, "ml", "models", "evaluation_results.json")
    results = None
    if os.path.exists(path):
        with open(path) as fh:
            results = json.load(fh)
    return render(request, "dashboard_admin/ml_analytics.html", {
        "results": results,
        "eval_summary": _evaluation_verdict(results) if results else None,
    })


# ── Transaction actions ─────────────────────────────────────────────
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from marketplace.services.transaction_service import (
    create_transaction, confirm_transaction, cancel_transaction, TransactionError,
)


@login_required
def start_transaction(request, pk):
    if not request.user.is_buyer:
        messages.error(request, "Only buyers can start transactions.")
        return redirect("listing_detail", pk=pk)
    if request.method != "POST":
        return redirect("listing_detail", pk=pk)
    qty = request.POST.get("quantity", "0")
    try:
        tx = create_transaction(request.user, pk, qty)
        messages.success(request,
            f"Request sent: {tx.quantity} {tx.listing.unit} of "
            f"{tx.listing.produce_name}. Total KSh {tx.total_amount:.2f}. "
            "Waiting for the farmer to confirm.")
        return redirect("dashboard")
    except TransactionError as e:
        messages.error(request, str(e))
        return redirect("listing_detail", pk=pk)


@login_required
def confirm_tx(request, tx_id):
    if request.method != "POST":
        return redirect("dashboard")
    try:
        confirm_transaction(request.user, tx_id)
        messages.success(request, "Transaction confirmed and stock updated.")
    except PermissionDenied as e:
        messages.error(request, str(e))
    except TransactionError as e:
        messages.error(request, str(e))
    return redirect("dashboard")


@login_required
def cancel_tx(request, tx_id):
    if request.method != "POST":
        return redirect("dashboard")
    try:
        cancel_transaction(request.user, tx_id)
        messages.success(request, "Transaction cancelled.")
    except PermissionDenied as e:
        messages.error(request, str(e))
    except TransactionError as e:
        messages.error(request, str(e))
    return redirect("dashboard")


# ── Farmer listing management + price estimate ──────────────────────
from decimal import Decimal, InvalidOperation
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse


@login_required
def create_listing(request):
    if not request.user.is_farmer:
        messages.error(request, "Only farmers can create listings.")
        return redirect("dashboard")

    if request.method == "POST":
        name = request.POST.get("produce_name", "").strip()
        location = request.POST.get("location", "").strip()
        try:
            # Decimal, not float: float("45.10") is not exactly 45.10 and
            # then fails the model's 2-decimal-place validation.
            quantity = Decimal(request.POST.get("quantity", "0").strip())
            price = Decimal(request.POST.get("asking_price", "0").strip())
            if not (quantity.is_finite() and price.is_finite()):
                raise ValueError
        except (InvalidOperation, ValueError):
            messages.error(request, "Enter valid numbers for quantity and price.")
            return redirect("create_listing")

        listing = ProduceListing(
            farmer=request.user, produce_name=name, quantity=quantity,
            asking_price=price, location=location or request.user.location,
            unit=request.POST.get("unit", "kg"),
            description=request.POST.get("description", "").strip(),
        )
        try:
            listing.full_clean()
            listing.save()
            messages.success(request, f"Listing created: {name}.")
            return redirect("dashboard")
        except Exception as e:
            msg = "; ".join(f"{k}: {v[0]}" for k, v in e.message_dict.items()) \
                  if hasattr(e, "message_dict") else str(e)
            messages.error(request, msg)
            return redirect("create_listing")

    # The dropdown reflects the crops the marketplace supports (same set the
    # seed data and recommender use), not only the crops the FAO price model
    # covers. For crops without price history the estimate endpoint returns
    # "No price history", which is expected.
    from marketplace.constants import SUPPORTED_PRODUCE
    produce_options = sorted(SUPPORTED_PRODUCE)
    return render(request, "farmer/create_listing.html",
                  {"produce_options": produce_options})


@login_required
def price_estimate(request):
    """AJAX endpoint: FAO-based estimated price for a produce type."""
    produce = request.GET.get("produce", "").strip()
    if not produce:
        return JsonResponse({"ok": False})
    try:
        from ml.recommendation.price_prediction import predict_price
        import datetime
        est = predict_price(produce, datetime.date.today().year)
        if est is None:
            return JsonResponse({"ok": False,
                "message": "No price history for this produce yet."})
        return JsonResponse({"ok": True, "estimate": est, "produce": produce})
    except Exception:
        return JsonResponse({"ok": False})


def _flag_transactions(limit=100):
    """Score recent transactions with the trained fraud model; return flagged ones."""
    try:
        from ml.recommendation.fraud_detection import score_transaction
    except Exception:
        return []
    txns = (Transaction.objects.select_related("buyer", "farmer", "listing")
            .order_by("-transaction_date")[:limit])
    if not txns:
        return []
    # precompute median price per produce for deviation feature
    from django.db.models import Count as _C
    from collections import defaultdict
    prices = defaultdict(list)
    for t in txns:
        if t.listing:
            prices[t.listing.produce_name].append(float(t.agreed_price))
    med = {k: (sorted(v)[len(v)//2] if v else 1) for k, v in prices.items()}
    buyer_freq = defaultdict(int)
    for t in txns:
        buyer_freq[t.buyer_id] += 1

    flagged = []
    for t in txns:
        produce = t.listing.produce_name if t.listing else ""
        m = med.get(produce, float(t.agreed_price) or 1)
        dev = abs(float(t.agreed_price) - m) / (m or 1)
        try:
            is_anom, score = score_transaction(
                amount=float(t.total_amount), quantity=float(t.quantity),
                price=float(t.agreed_price), price_deviation=dev,
                buyer_freq=buyer_freq[t.buyer_id])
        except FileNotFoundError:
            # Model not trained yet — show the dashboard without fraud review
            # rather than a 500. Run `python manage.py train_recommender`.
            return []
        if is_anom:
            flagged.append({
                "id": t.id,
                "buyer": t.buyer.get_full_name() or t.buyer.username,
                "farmer": t.farmer.get_full_name() or t.farmer.username,
                "produce": produce or "—",
                "quantity": t.quantity,
                "amount": t.total_amount,
                "score": round(score, 3),
            })
    return flagged
