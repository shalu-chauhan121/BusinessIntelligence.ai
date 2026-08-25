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

### Metric columns

**Nothing is hard-coded to one kind of business.** Column names are matched to KPI
*concepts*, so `revenue`, `net_sales`, `turnover` and `billed_amount` all mean the
same thing, and a hospital's `admissions`, `recovered_patients` and `bed_days` are
understood just as well as a retailer's `orders` and `stockout_events`.

What the system works out for each numeric column:

| It infers | From | Why it matters |
|---|---|---|
| semantic type | name tokens + the values | money, count, rate, duration, score |
| additivity | name + distribution | a **flow** is summed; a **stock** like `inventory_units` is averaged, because twelve month-end levels do not add up to an annual level |
| containment | the actual rows | `recovered ≤ discharges` is what makes recovery rate a real rate; `orders ≤ revenue` is arithmetic and means nothing |

Concepts the shipped library knows include revenue, cost, orders, customers,
units, marketing spend, fulfilled orders, stockouts, inventory, returns and
support tickets (general and retail), plus packs for healthcare, logistics,
subscription and manufacturing.

**A KPI only appears when your data can actually support it.** Where a concept
does not bind, the KPI is listed as unavailable with the reason — it is never
invented. The Business data page shows both lists for your file.

### Derived KPIs (computed for you, never uploaded)

Derived KPIs are proposed from combinations that are *semantically* valid, not
merely computable:

* `revenue` + `cost` → gross profit and gross margin %
* an outcome contained by a population → a rate (recovery rate, return rate,
  fulfilment rate, on-time delivery rate)
* money ÷ a population → unit economics (average order value, cost per admission)
* a duration → an average (average length of stay)

`revenue / cost_of_goods` is *not* proposed: cost is a component of revenue, so
the meaningful KPI for that pair is a margin, not a coverage multiple.

### Reviewing and overriding the definitions

Everything discovered is a **proposal**. On the **KPI contract** page you confirm
each KPI's granularity, edit any definition, add metrics your organisation uses
that the data cannot imply, resolve anything the system flagged as ambiguous, and
approve. Only approved KPIs are authoritative for the analysis.

Granularity matters more than it looks: two KPIs at different grains must not be
compared or aggregated together, and a rate must be recomputed from its numerator
and denominator at every level rather than averaged.

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
