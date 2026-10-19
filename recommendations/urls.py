from django.urls import path
from . import views

urlpatterns = [
    path("recommendations/", views.buyer_recommendations, name="my-recommendations"),
    path("recommendations/<str:buyer_id>/", views.buyer_recommendations,
         name="buyer-recommendations"),
]
