"""
FAO Kenya Producer Prices preprocessing for the price prediction module.

Source: FAOSTAT Producer Prices (PP), Area = Kenya, real official data
2010-2024 (annual), USD/tonne. This is genuine data — no fabrication.
"""
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                    "fao_kenya_prices.csv")

# Map FAO item names -> the produce names used in the marketplace
FAO_TO_MARKET = {
    "Beans, dry": "Beans",
    "Mangoes, guavas and mangosteens": "Mangoes",
    "Tea leaves": "Tea",
    "Potatoes": "Potatoes", "Rice": "Rice", "Tomatoes": "Tomatoes",
    "Wheat": "Wheat", "Cabbages": "Cabbage", "Bananas": "Bananas",
    "Avocados": "Avocado", "Oranges": "Oranges", "Sorghum": "Sorghum",
}

# USD/tonne -> KES/kg  (approx 1 USD = 130 KES; /1000 for tonne->kg)
USD_TO_KES = 130.0


def load_clean():
    df = pd.read_csv(DATA)
    df = df[df["Element"].str.contains("Producer Price", na=False)]
    df = df[["Item", "Year", "Value", "Unit"]].dropna(subset=["Value"])
    df = df[df["Value"] > 0]
    df["produce"] = df["Item"].map(FAO_TO_MARKET).fillna(df["Item"])
    # convert USD/tonne to KES/kg for marketplace realism
    df["price_kes_kg"] = (df["Value"] * USD_TO_KES / 1000).round(2)
    df["year"] = df["Year"].astype(int)
    return df[["produce", "year", "price_kes_kg"]].reset_index(drop=True)


if __name__ == "__main__":
    df = load_clean()
    print(f"Rows: {len(df)} | produce types: {df['produce'].nunique()} "
          f"| years: {df['year'].min()}-{df['year'].max()}")
    print("\nSample — Potatoes over time (real FAO data, KES/kg):")
    print(df[df["produce"]=="Potatoes"].to_string(index=False))
