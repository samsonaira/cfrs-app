"""
WFP monthly market-price preprocessing for the price prediction module.

Source: World Bank / WFP Real-Time Food Prices (RTFP) for Kenya, monthly,
233 markets, 2007-2026 (real data). Only maize and beans are covered by
this dataset — these are the staple crops for which granular monthly
market-level price data is publicly available. Prices are KES per kg.
This is genuine data; nothing is fabricated.
"""
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                    "wfp_monthly_prices.csv")

COVERED_CROPS = ["Maize", "Beans"]


def load_monthly():
    """Return long-format monthly prices: produce, region, market, year, month, price_kes_kg."""
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["price_kes_kg"])
    df = df[df["price_kes_kg"] > 0]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_monthly()
    print(f"Rows: {len(df)} | crops: {sorted(df['produce'].unique())} "
          f"| markets: {df['market'].nunique()} | {df['year'].min()}-{df['year'].max()}")
