"""
Middleware that keeps unverified accounts out of the application.

Email/password signups are inactive until verified (so they can never log in),
but a social login can create an authenticated session for an account whose
provider email was not verified. This middleware redirects any authenticated,
non-superuser user whose email is not yet verified to the verification page,
until they confirm their email with the one-time code.
"""
from django.shortcuts import redirect

EXEMPT_PREFIXES = (
    "/verify-email",
    "/logout",
    "/accounts",   # allauth callback/login endpoints
    "/static",
    "/admin",
)


class EmailVerifiedRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)
        if (user is not None and user.is_authenticated
                and not user.is_superuser
                and not getattr(user, "email_verified", True)):
            path = request.path
            if not any(path.startswith(p) for p in EXEMPT_PREFIXES):
                request.session["pending_verification_account_id"] = user.id
                return redirect("verify_email")
        return self.get_response(request)
