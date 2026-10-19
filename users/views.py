from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST

from .forms import RegistrationForm
from .models import FarmerProfile, BuyerProfile
from marketplace.models import ProduceListing, Transaction, RecommendationRecord


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "landing/home.html")


def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            if user.role == "FARMER":
                FarmerProfile.objects.create(account=user,
                    farm_location=user.location or "Unspecified")
            else:
                BuyerProfile.objects.create(account=user,
                    preferred_location=user.location or "")
            login(request, user)
            messages.success(request, f"Welcome to CFRS, {user.first_name}.")
            return redirect("dashboard")
    else:
        form = RegistrationForm(initial={"role": request.GET.get("role", "BUYER")})
    return render(request, "auth/register.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, username=email, password=password)
        if user:
            login(request, user)
            return redirect("dashboard")
        messages.error(request, "Incorrect email or password. Try again.")
    return render(request, "auth/login.html")


@require_POST  # a GET must not change state: any link or <img> could log a user out
def logout_view(request):
    logout(request)
    return redirect("home")


@login_required
def dashboard(request):
    u = request.user
    if u.is_administrator:
        return redirect("admin_dashboard")
    if u.is_farmer:
        listings = ProduceListing.objects.filter(farmer=u)
        pending = Transaction.objects.filter(farmer=u, status="PENDING")
        completed = Transaction.objects.filter(farmer=u, status="COMPLETED")
        return render(request, "farmer/dashboard.html", {
            "listings": listings,
            "active_count": listings.filter(status="ACTIVE").count(),
            "pending": pending, "completed_count": completed.count(),
        })
    # buyer
    txns = Transaction.objects.filter(buyer=u).select_related("farmer", "listing")
    return render(request, "buyer/dashboard.html", {
        "transactions": txns[:5],
        "tx_count": txns.count(),
        "profile": getattr(u, "buyer_profile", None),
    })
