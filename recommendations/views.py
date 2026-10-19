"""Recommendation REST API."""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from users.models import Account
from .services import get_recommendations


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def buyer_recommendations(request, buyer_id=None):
    """
    GET  /api/recommendations/            -> recommendations for logged-in buyer
    GET  /api/recommendations/<buyer_id>/ -> admin may query any buyer
    POST /api/recommendations/            -> buyer requests + records their own

    GET is read-only (a safe method must not change state). The audit snapshot
    is written only when a buyer POSTs for their OWN recommendations, and it
    replaces the previous snapshot, so the table cannot grow without bound.
    Object-level authorization: a buyer may only fetch their own
    recommendations; administrators may fetch any (read-only).
    """
    requester = request.user

    # Resolve target buyer
    if buyer_id:
        if not requester.is_administrator:
            return Response(
                {"success": False,
                 "error": {"code": "FORBIDDEN",
                           "message": "You may only view your own recommendations."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        target = Account.objects.filter(username=buyer_id,
                                        role=Account.Role.BUYER).first()
        if not target:
            return Response(
                {"success": False,
                 "error": {"code": "NOT_FOUND", "message": "Buyer not found."}},
                status=status.HTTP_404_NOT_FOUND,
            )
    else:
        if not requester.is_buyer:
            return Response(
                {"success": False,
                 "error": {"code": "FORBIDDEN",
                           "message": "Only buyers receive recommendations."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        target = requester

    # Persist only when the buyer POSTs for their OWN recommendations.
    # GET never writes; admin viewing another buyer never writes.
    persist = request.method == "POST" and target == requester
    try:
        mode, recs = get_recommendations(target, top_n=10, persist=persist)
    except Exception as exc:  # never leak a stack trace
        import logging
        logging.getLogger("cfrs").exception("Recommendation failure")
        return Response(
            {"success": False,
             "error": {"code": "ML_ERROR",
                       "message": "Could not generate recommendations right now."}},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return Response({
        "success": True,
        "buyer_id": target.username,
        "recommendation_mode": mode,
        "count": len(recs),
        "recommendations": recs,
    })
