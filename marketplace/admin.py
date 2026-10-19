from django.contrib import admin

from .models import ProduceListing, Transaction, RecommendationRecord


@admin.register(ProduceListing)
class ProduceListingAdmin(admin.ModelAdmin):
    list_display = ("produce_name", "farmer", "quantity", "unit", "asking_price",
                    "location", "status", "created_at")
    list_filter = ("status", "produce_name", "location")
    search_fields = ("produce_name", "location", "farmer__email")
    raw_id_fields = ("farmer",)


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "buyer", "farmer", "listing", "quantity", "agreed_price",
                    "total_amount", "status", "transaction_date")
    list_filter = ("status",)
    search_fields = ("buyer__email", "farmer__email")
    raw_id_fields = ("buyer", "farmer", "listing")
    readonly_fields = ("total_amount",)


@admin.register(RecommendationRecord)
class RecommendationRecordAdmin(admin.ModelAdmin):
    list_display = ("buyer", "farmer", "produce_type", "location", "hybrid_score",
                    "mode", "generated_at")
    list_filter = ("mode",)
    raw_id_fields = ("buyer", "farmer")
