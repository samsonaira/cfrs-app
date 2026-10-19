from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import Account, FarmerProfile, BuyerProfile


@admin.register(Account)
class AccountAdmin(UserAdmin):
    """UserAdmin so passwords are hashed and changed through the proper form."""
    list_display = ("email", "first_name", "role", "location", "is_staff", "is_active")
    list_filter = ("role", "is_staff", "is_active")
    search_fields = ("email", "username", "first_name", "last_name")
    ordering = ("email",)
    fieldsets = UserAdmin.fieldsets + (
        ("CFRS", {"fields": ("role", "phone_number", "location")}),
    )
    add_fieldsets = (
        (None, {"classes": ("wide",),
                "fields": ("email", "username", "role", "password1", "password2")}),
    )


@admin.register(FarmerProfile)
class FarmerProfileAdmin(admin.ModelAdmin):
    list_display = ("account", "farm_location", "farm_size_acres")
    search_fields = ("account__email", "farm_location")


@admin.register(BuyerProfile)
class BuyerProfileAdmin(admin.ModelAdmin):
    list_display = ("account", "buyer_type", "preferred_produce", "preferred_location")
    list_filter = ("buyer_type",)
    search_fields = ("account__email", "business_name")
