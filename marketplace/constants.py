"""
Canonical crop list the CFRS marketplace supports.

This is the set farmers can list and that the synthetic dataset and seed data
use, so the listing dropdown, the seeded listings and the recommender all agree.
It is kept identical to PRODUCE in ml/data/generate_dataset.py (a test enforces
that they stay in sync). The FAO price model only covers a subset of these
crops; for the rest the price estimate returns "No price history", which is
expected.
"""

SUPPORTED_PRODUCE = [
    "Maize", "Rice", "Beans", "Wheat", "Tomatoes", "Potatoes", "Onions",
    "Avocado", "Coffee", "Tea", "Bananas", "Cabbage", "Kales", "Mangoes",
]
