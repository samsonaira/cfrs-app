"""
Management command: seed the database with demo accounts, farmers, buyers,
listings and transactions derived from the synthetic ML dataset.

    python manage.py seed_demo_data

Idempotent: uses get_or_create so re-running does not create duplicates.
The seeded buyers' usernames match their dataset IDs so the trained
preference model aligns with the ORM accounts.
"""
import os
import secrets
import pandas as pd
import joblib

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction as db_tx

from users.models import Account, FarmerProfile, BuyerProfile
from marketplace.models import ProduceListing, Transaction


# Password for the low-privilege demo farmer/buyer fixtures only. Overridable
# via env for anyone who wants a different one; these accounts hold no
# privileges, so a shared demo password is acceptable for them.
DEMO_PASSWORD = os.environ.get("CFRS_DEMO_PASSWORD", "demo12345")
BASE_PRICE = {
    "Maize": 45, "Rice": 120, "Beans": 110, "Wheat": 50, "Tomatoes": 80,
    "Potatoes": 40, "Onions": 70, "Avocado": 30, "Coffee": 500, "Tea": 400,
    "Bananas": 35, "Cabbage": 25, "Kales": 30, "Mangoes": 60,
}


class Command(BaseCommand):
    help = "Seed demo farmers, buyers, listings and transactions."

    def add_arguments(self, parser):
        parser.add_argument("--buyers", type=int, default=40,
                            help="Number of buyers to seed (default 40)")
        parser.add_argument("--farmers", type=int, default=120,
                            help="Number of farmers to seed (default 120)")

    @db_tx.atomic
    def handle(self, *args, **opts):
        data_path = os.path.join(settings.BASE_DIR, "ml", "data", "transactions.csv")
        if not os.path.exists(data_path):
            self.stdout.write(self.style.ERROR(
                "Dataset not found. Run: python ml/data/generate_dataset.py"))
            return

        df = pd.read_csv(data_path)
        df["produce_type"] = df["produce_type"].str.title()
        df["location"] = df["location"].str.title()

        n_buyers, n_farmers = opts["buyers"], opts["farmers"]

        # --- Demo role accounts -------------------------------------------
        self._demo_accounts()

        # --- Farmers (with an ACTIVE listing each) ------------------------
        farmer_ids = df["farmer_id"].value_counts().head(n_farmers).index.tolist()
        created_f = 0
        for fid in farmer_ids:
            rows = df[df.farmer_id == fid]
            loc = rows["location"].mode()[0]
            acct, made = Account.objects.get_or_create(
                username=fid,
                defaults=dict(email=f"{fid.lower()}@farm.ke", role="FARMER",
                              first_name=f"Farmer {fid}", location=loc,
                              email_verified=True),
            )
            if made:
                acct.set_password(DEMO_PASSWORD); acct.save()
                created_f += 1
            FarmerProfile.objects.get_or_create(
                account=acct, defaults=dict(farm_location=loc, farm_size_acres=5))
            # one active listing per top produce this farmer sells
            prod = rows["produce_type"].mode()[0]
            ProduceListing.objects.get_or_create(
                farmer=acct, produce_name=prod, status="ACTIVE",
                defaults=dict(quantity=500, unit="kg",
                              asking_price=BASE_PRICE.get(prod, 50), location=loc),
            )

        # --- Buyers -------------------------------------------------------
        buyer_ids = df["buyer_id"].value_counts().head(n_buyers).index.tolist()
        created_b = 0
        for bid in buyer_ids:
            rows = df[df.buyer_id == bid]
            top_prod = rows["produce_type"].mode()[0]
            top_loc = rows["location"].mode()[0]
            acct, made = Account.objects.get_or_create(
                username=bid,
                defaults=dict(email=f"{bid.lower()}@buyer.ke", role="BUYER",
                              first_name=f"Buyer {bid}", location=top_loc,
                              email_verified=True),
            )
            if made:
                acct.set_password(DEMO_PASSWORD); acct.save()
                created_b += 1
            BuyerProfile.objects.get_or_create(
                account=acct,
                defaults=dict(business_name=f"{bid} Trading",
                              preferred_produce=top_prod, preferred_location=top_loc))

        # --- A few completed transactions for realism ---------------------
        seeded_farmers = set(farmer_ids)
        seeded_buyers = set(buyer_ids)
        sample = df[(df.buyer_id.isin(seeded_buyers)) &
                    (df.farmer_id.isin(seeded_farmers)) &
                    (df.status == "COMPLETED")].head(300)
        tx_created = 0
        for _, row in sample.iterrows():
            buyer = Account.objects.filter(username=row.buyer_id).first()
            farmer = Account.objects.filter(username=row.farmer_id).first()
            listing = ProduceListing.objects.filter(farmer=farmer).first()
            if not (buyer and farmer and listing):
                continue
            _, made = Transaction.objects.get_or_create(
                buyer=buyer, farmer=farmer, listing=listing,
                quantity=row.quantity, agreed_price=row.price,
                defaults=dict(status="COMPLETED"),
            )
            tx_created += made

        self.stdout.write(self.style.SUCCESS(
            f"\nSeed complete:"
            f"\n  Farmers:      {Account.objects.filter(role='FARMER').count()} "
            f"(+{created_f} new)"
            f"\n  Buyers:       {Account.objects.filter(role='BUYER').count()} "
            f"(+{created_b} new)"
            f"\n  Active lists: {ProduceListing.objects.filter(status='ACTIVE').count()}"
            f"\n  Transactions: {Transaction.objects.count()} (+{tx_created} new)"))
        self.stdout.write("\nDemo logins (non-privileged, password: %s):" % DEMO_PASSWORD)
        self.stdout.write("  demo_farmer@cfrs.ke / demo_buyer@cfrs.ke")
        if self.admin_password:
            self.stdout.write(self.style.WARNING(
                "\nAdmin superuser demo_admin@cfrs.ke password (%s):" % self.admin_password_source))
            self.stdout.write(self.style.WARNING("  %s" % self.admin_password))
            self.stdout.write("  Record it now — it is not stored anywhere and not printed again.")
        else:
            self.stdout.write("\nAdmin superuser demo_admin@cfrs.ke already existed; password unchanged.")

    def _demo_accounts(self):
        """Create the three named demo accounts.

        The administrator is a superuser, so it never gets the shared demo
        password. It uses CFRS_ADMIN_PASSWORD if set, otherwise a fresh random
        password generated here and printed once at the end of the run — it is
        never hard-coded or published.
        """
        self.admin_password = None       # shown once if we create the admin
        self.admin_password_source = None
        specs = [
            ("demo_farmer", "demo_farmer@cfrs.ke", "FARMER"),
            ("demo_buyer", "demo_buyer@cfrs.ke", "BUYER"),
            ("demo_admin", "demo_admin@cfrs.ke", "ADMINISTRATOR"),
        ]
        for username, email, role in specs:
            acct, made = Account.objects.get_or_create(
                username=username,
                defaults=dict(email=email, role=role, email_verified=True,
                              first_name=username.replace("demo_", "Demo ").title()),
            )
            if made:
                if role == "ADMINISTRATOR":
                    env_pw = os.environ.get("CFRS_ADMIN_PASSWORD")
                    self.admin_password = env_pw or secrets.token_urlsafe(12)
                    self.admin_password_source = "env CFRS_ADMIN_PASSWORD" if env_pw \
                        else "randomly generated (shown once below)"
                    acct.set_password(self.admin_password)
                    acct.is_staff = True
                    acct.is_superuser = True
                else:
                    acct.set_password(DEMO_PASSWORD)
                acct.save()
            if role == "FARMER":
                FarmerProfile.objects.get_or_create(
                    account=acct, defaults=dict(farm_location="Kiambu"))
            elif role == "BUYER":
                BuyerProfile.objects.get_or_create(
                    account=acct, defaults=dict(business_name="Demo Retail",
                                                preferred_produce="Maize",
                                                preferred_location="Kiambu"))
