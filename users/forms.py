from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Account


class RegistrationForm(UserCreationForm):
    ROLE_CHOICES = [("BUYER", "Buyer"), ("FARMER", "Farmer")]
    role = forms.ChoiceField(choices=ROLE_CHOICES)
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=60, required=True)
    location = forms.CharField(max_length=120, required=False)

    class Meta:
        model = Account
        fields = ["first_name", "email", "location", "role", "password1", "password2"]

    def clean_email(self):
        email = self.cleaned_data["email"]
        if Account.objects.filter(email=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.username = self.cleaned_data["email"]  # email as username
        user.role = self.cleaned_data["role"]
        user.first_name = self.cleaned_data["first_name"]
        user.location = self.cleaned_data.get("location", "")
        if commit:
            user.save()
        return user
