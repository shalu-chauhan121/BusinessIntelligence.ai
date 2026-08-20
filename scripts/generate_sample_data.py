"""
Deterministic generator for the BusinessIntelligence.ai demo dataset.

Produces sample_data/business_metrics_sample.csv : a weekly, dimensioned
business fact table spanning 2023-01 .. 2026-06 (14 quarters).

A scenario is deliberately planted in Q2-2026 so that the four-stage
investigation has something real to find:

  * Demand erosion in the North region begins in early April 2026
    (customers/orders fall first, price roughly flat).
  * A supply disruption hits Product A from 11-May-2026 to 15-Jun-2026
    (inventory collapses, stockouts spike, fulfilment rate drops).

The supply disruption is therefore a *real* contributing factor whose
supporting evidence is strong -- but it starts ~5 weeks AFTER the revenue
decline began. That temporal contradiction is what the CONTEST stage is
built to surface. Nothing about this is hard-coded in the engines; the
engines only ever see the CSV.
"""
import csv
import math
import os
import random
from datetime import date, timedelta

SEED = 20260820
random.seed(SEED)

OUT = os.path.join(os.path.dirname(__file__), "..", "sample_data", "business_metrics_sample.csv")

REGIONS = {"North": 1.00, "South": 0.82, "East": 0.71, "West": 0.93}
PRODUCTS = {  # name -> (demand weight, unit price, unit cost)
    "Product A": (1.00, 268.0, 151.0),
    "Product B": (0.74, 189.0, 118.0),
    "Product C": (0.55, 412.0, 260.0),
}
CHANNELS = {"Online": 1.00, "Retail Partner": 0.72}
SEGMENTS = {"Enterprise": 1.00, "SMB": 0.66}

START = date(2023, 1, 2)          # a Monday
END = date(2026, 6, 29)

# --- scenario windows -------------------------------------------------------
DEMAND_EROSION_START = date(2026, 4, 6)     # North demand starts slipping
EAST_EROSION_START = date(2026, 5, 18)      # competitor expands into East later
SUPPLY_START = date(2026, 5, 4)             # supply disruption begins
SUPPLY_END = date(2026, 6, 22)


def month_seasonality(m: int) -> float:
    """Mild, repeating annual seasonality (peaks in Nov/Dec, troughs in Feb)."""
    return 1.0 + 0.030 * math.sin((m - 4.0) / 12.0 * 2 * math.pi)


def weeks(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=7)


def jitter(sd: float) -> float:
    return math.exp(random.gauss(0.0, sd))


rows = []
for d in weeks(START, END):
    week_index = (d - START).days / 7.0
    trend = 1.0 + 0.0015 * week_index          # slow organic growth
    season = month_seasonality(d.month)

    for region, rw in REGIONS.items():
        for product, (pw, price0, cost0) in PRODUCTS.items():
            for channel, cw in CHANNELS.items():
                for segment, sw in SEGMENTS.items():

                    demand_mult = 1.0
                    price_mult = 1.0
                    fill_rate = 1.0
                    inventory_mult = 1.0
                    stockout_mult = 1.0

                    # 1) Competitive demand erosion: North first, East later.
                    if region == "North" and d >= DEMAND_EROSION_START:
                        w = (d - DEMAND_EROSION_START).days / 7.0
                        ramp = min(1.0, 0.30 + 0.09 * w)
                        depth = 0.50 if product == "Product A" else 0.42
                        demand_mult *= 1.0 - depth * ramp
                        if channel == "Online":
                            price_mult *= 1.0 - 0.030 * ramp   # discounting to defend share
                    if region == "East" and d >= EAST_EROSION_START:
                        w = (d - EAST_EROSION_START).days / 7.0
                        ramp = min(1.0, 0.35 + 0.11 * w)
                        depth = 0.36 if product == "Product A" else 0.30
                        demand_mult *= 1.0 - depth * ramp

                    # 2) Supply disruption on Product A (worst in North)
                    if product == "Product A" and SUPPLY_START <= d <= SUPPLY_END:
                        severity = 1.0 if region == "North" else 0.90
                        fill_rate = 1.0 - 0.45 * severity
                        inventory_mult = 1.0 - 0.62 * severity
                        stockout_mult = 1.0 + 9.0 * severity

                    base_units = 42.0 * rw * pw * cw * sw
                    units_demanded = base_units * trend * season * demand_mult * jitter(0.055)
                    units_sold = units_demanded * fill_rate

                    price = price0 * (1.0 + 0.0004 * week_index) * price_mult * jitter(0.012)
                    revenue = units_sold * price

                    orders = max(1.0, units_sold / (2.6 * (1.35 if segment == "Enterprise" else 1.0)))
                    customers = max(1.0, orders * 0.78 * jitter(0.03))
                    fulfilled = orders * (fill_rate * 0.985 + 0.015 * fill_rate)
                    stockouts = max(0.0, orders * 0.045 * stockout_mult * jitter(0.25))
                    inventory = base_units * 5.4 * season * inventory_mult * jitter(0.05)
                    cogs = units_sold * cost0 * jitter(0.01)
                    marketing = revenue * (0.093 if channel == "Online" else 0.052) * jitter(0.08)
                    returns = orders * 0.062 * (1.6 if fill_rate < 1.0 else 1.0) * jitter(0.2)
                    tickets = orders * 0.145 * (2.1 if fill_rate < 1.0 else 1.0) * jitter(0.15)

                    rows.append({
                        "date": d.isoformat(),
                        "region": region,
                        "product": product,
                        "channel": channel,
                        "segment": segment,
                        "revenue": round(revenue, 2),
                        "units_sold": int(round(units_sold)),
                        "orders": int(round(orders)),
                        "customers": int(round(customers)),
                        "fulfilled_orders": int(round(fulfilled)),
                        "stockout_events": int(round(stockouts)),
                        "inventory_units": int(round(inventory)),
                        "cost_of_goods": round(cogs, 2),
                        "marketing_spend": round(marketing, 2),
                        "returns": int(round(returns)),
                        "support_tickets": int(round(tickets)),
                    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows):,} rows -> {os.path.abspath(OUT)}")
