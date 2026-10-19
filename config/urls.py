from django.contrib import admin
from django.urls import path, include
from users import views as uv
from marketplace import views as mv

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", uv.home, name="home"),
    path("register/", uv.register, name="register"),
    path("login/", uv.login_view, name="login"),
    path("logout/", uv.logout_view, name="logout"),
    path("dashboard/", uv.dashboard, name="dashboard"),
    path("marketplace/", mv.marketplace, name="marketplace"),
    path("marketplace/<int:pk>/", mv.listing_detail, name="listing_detail"),
    path("listings/new/", mv.create_listing, name="create_listing"),
    path("api/price-estimate/", mv.price_estimate, name="price_estimate"),
    path("marketplace/<int:pk>/buy/", mv.start_transaction, name="start_transaction"),
    path("tx/<int:tx_id>/confirm/", mv.confirm_tx, name="confirm_tx"),
    path("tx/<int:tx_id>/cancel/", mv.cancel_tx, name="cancel_tx"),
    path("admin-dashboard/", mv.admin_dashboard, name="admin_dashboard"),
    path("ml-analytics/", mv.ml_analytics, name="ml_analytics"),
    path("api/", include("recommendations.urls")),
]
