"""
Synthetic agricultural transaction dataset generator for CFRS development.

IMPORTANT: This is clearly-labelled SYNTHETIC development data. It is NOT
real market data. It is engineered to contain genuine latent structure
(buyer segments with produce + location affinities) so that collaborative
filtering has real patterns to discover — the CF results are therefore
meaningful, not noise.
"""
import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

PRODUCE = ["Maize", "Rice", "Beans", "Wheat", "Tomatoes", "Potatoes", "Onions",
           "Avocado", "Coffee", "Tea", "Bananas", "Cabbage", "Kales", "Mangoes"]

COUNTIES = ["Nairobi", "Kiambu", "Nakuru", "Eldoret", "Mombasa", "Kisumu",
            "Meru", "Nyeri", "Machakos", "Kakamega", "Kericho", "Bungoma",
            "Kitui", "Embu", "Trans-Nzoia"]

N_FARMERS = 500
N_BUYERS = 200
N_TRANSACTIONS = 15000

# Base price per kg per produce (KES) — realistic ranges
BASE_PRICE = {
    "Maize": 45, "Rice": 120, "Beans": 110, "Wheat": 50, "Tomatoes": 80,
    "Potatoes": 40, "Onions": 70, "Avocado": 30, "Coffee": 500, "Tea": 400,
    "Bananas": 35, "Cabbage": 25, "Kales": 30, "Mangoes": 60,
}


def build_farmers():
    """Each farmer specialises in 1-2 produce types and sits in one county."""
    farmers = {}
    for i in range(N_FARMERS):
        fid = f"F{i:04d}"
        county = random.choice(COUNTIES)
        specialties = random.sample(PRODUCE, k=random.choice([1, 1, 2]))
        farmers[fid] = {"county": county, "produce": specialties}
    return farmers


def build_buyers():
    """
    Each buyer belongs to a segment with a STRONG produce affinity and a
    preferred region. This is the latent structure CF will learn.
    """
    buyers = {}
    for i in range(N_BUYERS):
        bid = f"B{i:04d}"
        # a primary produce preference (weighted) + secondary
        primary = random.choice(PRODUCE)
        secondary = random.choice([p for p in PRODUCE if p != primary])
        preferred_region = random.choice(COUNTIES)
        buyers[bid] = {
            "primary": primary,
            "secondary": secondary,
            "region": preferred_region,
        }
    return buyers


def generate():
    farmers = build_farmers()
    buyers = build_buyers()
    farmer_ids = list(farmers.keys())

    # Index farmers by produce for realistic matching
    by_produce = {}
    for fid, f in farmers.items():
        for p in f["produce"]:
            by_produce.setdefault(p, []).append(fid)

    # Give each buyer a small pool of trusted farmers they return to.
    # This is what creates repeat-purchase structure for CF to learn.
    for bid, buyer in buyers.items():
        pool = set()
        for prod in (buyer["primary"], buyer["secondary"]):
            cands = by_produce.get(prod, farmer_ids)
            region_c = [f for f in cands if farmers[f]["county"] == buyer["region"]]
            src = region_c if len(region_c) >= 3 else cands
            pool.update(random.sample(src, k=min(4, len(src))))
        # sorted(): set order of strings changes between Python runs (hash
        # randomisation), which made the "fixed-seed" dataset differ every time.
        buyer["trusted"] = sorted(pool)

    rows = []
    start = datetime(2023, 1, 1)
    for t in range(N_TRANSACTIONS):
        bid = random.choice(list(buyers.keys()))
        buyer = buyers[bid]

        # Buyer picks produce according to their affinity (70% primary/secondary)
        r = random.random()
        if r < 0.55:
            produce = buyer["primary"]
        elif r < 0.75:
            produce = buyer["secondary"]
        else:
            produce = random.choice(PRODUCE)

        # 65% of the time buy from a trusted farmer (repeat purchase);
        # otherwise explore. This yields learnable CF structure.
        trusted_for_produce = [f for f in buyer["trusted"]
                               if produce in farmers[f]["produce"]]
        if trusted_for_produce and random.random() < 0.65:
            fid = random.choice(trusted_for_produce)
        else:
            candidates = by_produce.get(produce, farmer_ids)
            region_match = [f for f in candidates
                            if farmers[f]["county"] == buyer["region"]]
            if region_match and random.random() < 0.6:
                fid = random.choice(region_match)
            else:
                fid = random.choice(candidates)

        location = farmers[fid]["county"]
        qty = random.choice([50, 100, 200, 300, 500, 800, 1000])
        base = BASE_PRICE[produce]
        price = round(base * random.uniform(0.85, 1.15), 2)
        date = start + timedelta(days=random.randint(0, 730))
        status = random.choices(["COMPLETED", "CANCELLED"], weights=[92, 8])[0]

        rows.append({
            "transaction_id": f"T{t:05d}",
            "buyer_id": bid,
            "farmer_id": fid,
            "produce_type": produce,
            "quantity": qty,
            "unit": "kg",
            "price": price,
            "location": location,
            "transaction_date": date.strftime("%Y-%m-%d"),
            "status": status,
        })

    return rows, farmers, buyers


def main():
    here = os.path.dirname(__file__)
    out = os.path.join(here, "transactions.csv")
    rows, farmers, buyers = generate()
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} transactions to {out}")
    print(f"Farmers: {len(farmers)} | Buyers: {len(buyers)} | Produce: {len(PRODUCE)} | Counties: {len(COUNTIES)}")
    completed = sum(1 for r in rows if r["status"] == "COMPLETED")
    print(f"Completed: {completed} | Cancelled: {len(rows)-completed}")


if __name__ == "__main__":
    main()
