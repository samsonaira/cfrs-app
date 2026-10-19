from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST

from . import otp
from .forms import RegistrationForm
from .models import Account, FarmerProfile, BuyerProfile
from marketplace.models import ProduceListing, Transaction, RecommendationRecord

PENDING_SESSION_KEY = "pending_verification_account_id"


def _mask_email(email):
    """s***@example.com — enough to recognise, not enough to expose."""
    try:
        local, domain = email.split("@", 1)
    except ValueError:
        return email
    shown = local[0] if local else ""
    return f"{shown}***@{domain}"


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "landing/home.html")


def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # New accounts start unverified and inactive: they cannot log in
            # until the email is confirmed with a one-time code.
            user.is_active = False
            user.email_verified = False
            user.save(update_fields=["is_active", "email_verified"])

            if user.role == "FARMER":
                FarmerProfile.objects.create(
                    account=user, farm_location=user.location or "Unspecified")
            else:
                BuyerProfile.objects.create(
                    account=user, preferred_location=user.location or "")

            ok, msg = otp.create_and_send(user, force=True)
            request.session[PENDING_SESSION_KEY] = user.id
            if ok:
                messages.info(request, "We've sent a 6-digit code to your email. "
                                       "Enter it below to verify your account.")
            else:
                messages.error(request, msg)
            return redirect("verify_email")
    else:
        form = RegistrationForm(initial={"role": request.GET.get("role", "BUYER")})
    return render(request, "auth/register.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, username=email, password=password)
        if user:
            # Even an active account must have a verified email (defence in depth).
            if not user.email_verified and not user.is_superuser:
                request.session[PENDING_SESSION_KEY] = user.id
                messages.info(request, "Please verify your email before logging in. "
                                       "Enter the code we sent you.")
                return redirect("verify_email")
            login(request, user)
            return redirect("dashboard")

        # authenticate() returns None for inactive (unverified) users too, so
        # distinguish "wrong password" from "correct password, not yet verified".
        acct = Account.objects.filter(email=email).first()
        if (acct and not acct.email_verified and not acct.is_superuser
                and acct.check_password(password or "")):
            request.session[PENDING_SESSION_KEY] = acct.id
            messages.info(request, "Please verify your email before logging in. "
                                   "Enter the code we sent you.")
            return redirect("verify_email")

        messages.error(request, "Incorrect email or password. Try again.")
    return render(request, "auth/login.html")


def verify_email(request):
    account_id = request.session.get(PENDING_SESSION_KEY)
    if not account_id:
        messages.error(request, "No verification in progress. Please register or log in.")
        return redirect("login")

    account = Account.objects.filter(id=account_id).first()
    if account is None:
        request.session.pop(PENDING_SESSION_KEY, None)
        return redirect("register")
    if account.email_verified:
        request.session.pop(PENDING_SESSION_KEY, None)
        messages.success(request, "Your email is already verified. Please log in.")
        return redirect("login")

    if request.method == "POST":
        ok, msg = otp.verify(account, request.POST.get("code", ""))
        if ok:
            request.session.pop(PENDING_SESSION_KEY, None)
            login(request, account, backend="django.contrib.auth.backends.ModelBackend")
            messages.success(request, "Email verified. Welcome to CFRS!")
            return redirect("dashboard")
        messages.error(request, msg)

    return render(request, "auth/verify_email.html", {
        "masked_email": _mask_email(account.email),
        "resend_cooldown": otp._resend_cooldown_seconds(),
    })


def social_start(request, provider):
    """
    Entry point for the 'Continue with Google/GitHub' buttons. Records the role
    chosen on the registration page (default Buyer) and hands off to allauth's
    provider login, which starts the OAuth flow.
    """
    if provider not in ("google", "github"):
        return redirect("login")
    role = request.GET.get("role", "BUYER")
    request.session["social_signup_role"] = role if role in ("FARMER", "BUYER") else "BUYER"
    return redirect(f"/accounts/{provider}/login/")


def resend_code(request):
    if request.method != "POST":
        return redirect("verify_email")
    account_id = request.session.get(PENDING_SESSION_KEY)
    account = Account.objects.filter(id=account_id).first() if account_id else None
    if account is None or account.email_verified:
        return redirect("login")
    ok, msg = otp.create_and_send(account)
    (messages.info if ok else messages.error)(request, msg)
    return redirect("verify_email")


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
