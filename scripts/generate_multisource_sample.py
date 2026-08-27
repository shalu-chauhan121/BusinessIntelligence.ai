"""
Split `sample_data/business_metrics_sample.csv` into three heterogeneous sources
for the reconciliation demo.

The whole point of the design is that there is **nothing to map**: every file
below uses the same field names, because within one organisation `revenue` means
what the KPI Contract says it means regardless of which system emitted the row.
So there is no bindings file, no field map, and nothing for the user to
configure — they drop three CSVs and get one reconciled view.

The three sources are shaped to exercise every reconciliation mode at once:

  sales.csv       revenue/units/orders/customers for North + South
                  -> PARTITION with finance on the regions it does not carry
  finance.csv     revenue for East + West (partition), plus cost_of_goods and
                  marketing_spend for ALL regions (complementary), plus revenue
                  for North that AGREES with sales (corroboration)
  operations.csv  fulfilled_orders/stockout_events for all regions, and it lags
                  a week behind -> the honest "this source is behind" case

Deterministic: every value is taken from the source file or a fixed transform of
it. Re-running reproduces byte-identical output.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "sample_data" / "business_metrics_sample.csv"
OUT_DIR = ROOT / "sample_data" / "multisource"

# Operations is deliberately a week behind the others, so the demo shows a real
# lagging feed rather than an invented one.
OPS_LAG_WEEKS = 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_CSV, parse_dates=["date"])

    # Roll the row-level (date, region, product, channel, segment) grain up to
    # the (date, region, product) business grain the three sources agree on.
    base = df.groupby(["date", "region", "product"], as_index=False).agg(
        revenue=("revenue", "sum"),
        units_sold=("units_sold", "sum"),
        orders=("orders", "sum"),
        customers=("customers", "sum"),
        cost_of_goods=("cost_of_goods", "sum"),
        marketing_spend=("marketing_spend", "sum"),
        fulfilled_orders=("fulfilled_orders", "sum"),
        stockout_events=("stockout_events", "sum"),
    )
    base = base.rename(columns={"product": "product_id"})
    base["date"] = base["date"].dt.strftime("%Y-%m-%d")

    west = base["region"].isin(["East", "West"])
    east = ~west

    # -- Sales: the commercial systems for North and South ------------------
    sales = base.loc[east, ["date", "region", "product_id",
                            "revenue", "units_sold", "orders", "customers"]]
    sales.to_csv(OUT_DIR / "sales.csv", index=False)

    # -- Finance: revenue for the regions Sales does not carry (partition),
    #    cost/marketing everywhere (complementary), and a corroborating copy of
    #    North's revenue that agrees exactly with Sales.
    finance_rev = base.loc[west, ["date", "region", "product_id", "revenue",
                                  "cost_of_goods", "marketing_spend"]]
    finance_other = base.loc[east, ["date", "region", "product_id",
                                    "cost_of_goods", "marketing_spend"]].copy()
    corroborating = base.loc[base["region"] == "North",
                             ["date", "region", "product_id", "revenue"]]
    finance = pd.concat([finance_rev, finance_other, corroborating], ignore_index=True)
    finance = finance.groupby(["date", "region", "product_id"], as_index=False).first()
    finance.to_csv(OUT_DIR / "finance.csv", index=False)

    # -- Operations: fulfilment signals everywhere, a week behind -----------
    last = pd.to_datetime(base["date"]).max()
    cutoff = (last - pd.Timedelta(weeks=OPS_LAG_WEEKS)).strftime("%Y-%m-%d")
    ops = base.loc[pd.to_datetime(base["date"]) <= cutoff,
                   ["date", "region", "product_id", "fulfilled_orders", "stockout_events"]]
    ops.to_csv(OUT_DIR / "operations.csv", index=False)

    print(f"sales.csv      {len(sales):>5} rows   North/South revenue + volume")
    print(f"finance.csv    {len(finance):>5} rows   East/West revenue, cost everywhere, "
          f"North revenue corroborating")
    print(f"operations.csv {len(ops):>5} rows   fulfilment, lagging to {cutoff}")
    print("\nNo bindings file: the sources already share field names, which is the point.")
    print("Upload all three together on the Business data page.")


if __name__ == "__main__":
    main()
