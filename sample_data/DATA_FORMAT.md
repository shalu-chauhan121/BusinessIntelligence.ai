# Input data format

`BusinessIntelligence.ai` analyses **your** data. Nothing in the product is hard-coded to
the demo company — every number you see in the app is computed from the file you upload.

There are two kinds of input.

---

## 1. Structured data — the business fact table (CSV)

One CSV. One row per **time period × business dimensions**. Weekly rows are recommended
(daily and monthly also work). Upload it on the **Data** page.

### Required columns

| Column | Type | Notes |
|---|---|---|
| `date` | `YYYY-MM-DD` | Period start date. Weekly rows = one row per Monday. |
| at least one metric column | number | e.g. `revenue` |

### Dimension columns (optional, but this is what makes root-cause analysis work)

| Column | Example values |
|---|---|
| `region` | North, South, East, West |
| `product` | Product A, Product B |
| `channel` | Online, Retail Partner |
| `segment` | Enterprise, SMB |

Any other text column is treated as an additional dimension automatically.

### Metric columns recognised out of the box

| Column | Aggregation | Used for |
|---|---|---|
| `revenue` | sum | headline KPI, driver decomposition |
| `units_sold` | sum | volume vs price decomposition |
| `orders` | sum | demand signal |
| `customers` | sum | demand signal, CAC |
| `fulfilled_orders` | sum | fulfilment rate |
| `stockout_events` | sum | stockout rate (supply hypotheses) |
| `inventory_units` | mean (stock level) | inventory cover (supply hypotheses) |
| `cost_of_goods` | sum | gross margin |
| `marketing_spend` | sum | CAC, marketing efficiency |
| `returns` | sum | quality signal |
| `support_tickets` | sum | quality / experience signal |

Unknown numeric columns are still ingested and summed — they appear as extra KPIs and can
be used as evidence, they just have no pre-built hypothesis attached to them.

### Derived KPIs (computed for you, never uploaded)

`gross_margin_pct`, `avg_order_value`, `avg_selling_price`, `fulfillment_rate`,
`stockout_rate`, `return_rate`, `customer_acquisition_cost`, `units_per_order`,
`tickets_per_1k_orders`.

### Minimum history

The Observe stage compares the selected quarter against its own history. Provide at least
**5 quarters**; **8+ quarters** enables the seasonal (year-on-year) baseline. With less
history the app still runs but will tell you the significance test is weakly powered.

### Template & sample

- `business_metrics_TEMPLATE.csv` — headers plus three example rows.
- `business_metrics_sample.csv` — 8,784 rows, 14 quarters, a full demo company.

---

## 2. Unstructured data — business documents

Upload `.md`, `.txt`, `.pdf` or `.csv` files on the **Documents** page: operations reports,
customer feedback, market intelligence, management commentary, meeting notes.

These are chunked and indexed for retrieval. The Investigate and Contest stages quote them
as evidence **with the document name and chunk shown**, so every textual claim in the final
story is traceable back to a document you uploaded.

`sample_data/documents/` contains seven documents for the demo company.

---

## What the system will not do

It will not invent numbers that are not in your file. Every figure rendered in the UI is
produced by the deterministic analysis layer (pandas/NumPy) from your uploaded rows. The
language model is only allowed to reason over, and write about, numbers the analysis layer
computed.
