# How drivers are identified and ranked

Every number in this document was produced by the deterministic analysis layer
(pandas/NumPy) from `sample_data/business_metrics_sample.csv`. No language model is
involved at any point in this document. Reproduce it with:

```bash
python3 scripts/generate_sample_data.py     # data + ground truth
python3 scripts/validate_rca.py             # the scorecard in §9
```

---

## 1. What counts as a driver

A dashboard answers *"revenue fell 17%"*. The next question is *"driven by what?"*, and
the naive answer is wrong in a specific, repeatable way.

Rank the segments of a declining business by how much of the decline each carries, and the
answer is always the biggest segment. On this dataset `Enterprise` is 60.2% of the business
and carries 60.5% of the decline. That is not a finding — it is arithmetic. Enterprise
carries 60% of *everything*.

So a driver is not "the part that carries the most change". A driver is **a part that
carries more of the change than being that size would imply, in a way that is large
against its own history, that persisted, and that lies on an axis the movement is actually
shaped by.** Four separate claims, four separate tests, none of them sufficient alone.

| Test | Catches | Fails alone because |
|---|---|---|
| **Magnitude** — share of the movement | the parts that matter commercially | the largest segment always wins |
| **Over-index** — share of movement ÷ share of business | segments moving out of proportion to size | a small segment clears it on noise |
| **Significance** — robust z against the member's *own* history | moves that are unusual for that member | a reliably volatile member is still not a driver |
| **Persistence** — weeks outside its own band | sustained problems, not one-week spikes | a factor that started late is still real |
| **Axis weight** — how much its dimension explains the shape | suppresses dimensions the change isn't shaped by | — |

---

## 2. Decomposition: how much sits where

### 2.1 Additive KPIs — exact contribution

For a KPI that sums (`revenue`, `orders`, `units_sold`), the decomposition over one
dimension is exact:

```
contribution_pct(m) = ( value_cur(m) − value_base(m) ) / ( total_cur − total_base ) × 100
```

Member deltas sum to the total delta and contributions sum to 100% by construction —
asserted in [`test_observe.py`](../backend/tests/test_observe.py) (`test_deltas_sum_to_total_delta`,
`test_contributions_sum_to_100`).

### 2.2 Ratio KPIs — the rate/mix split

For a ratio (`gross_margin_pct`, `fulfillment_rate`), a contribution figure alone hides
which of two entirely different business problems occurred. The KPI can fall because
members got worse, or because the *mix* shifted toward weaker members. So the delta is
split:

```
ΔR = Σ w_i,cur·(r_i,cur − r_i,base)     rate effect  — members got worse
   + Σ r_i,base·(w_i,cur − w_i,base)    mix  effect  — the mix moved
```

where `r_i` is member *i*'s own rate and `w_i` its share of the denominator. Both effects
are emitted per member under `effects`. A rate is always recomputed from its numerator and
denominator at every level — never averaged from member rates, which would be an average of
averages ([`test_ratio_kpi_is_not_an_average_of_averages`](../backend/tests/test_observe.py)).

Implementation: [`decompose_dimension`](../backend/app/engines/observe.py).

### 2.3 The problem neither of those solves

Both decompositions run **one dimension at a time**, independently. Their results do not
compose. On this dataset:

- `region = North` carries 64.5% of the decline
- `product = Product A` carries 78.3% of the decline

Read side by side, those imply 142.8% of a decline that is only 100% large. They overlap,
because the supply disruption lives in the *cell where they intersect*. Nothing in a
per-dimension decomposition can say so — there is no interaction term, and the parts do not
sum to the whole.

---

## 3. Which axis is the story on — exact Shapley

### 3.1 The construction

Cross every dimension into a cell grid (here `region × product × channel × segment`
= 48 cells). For each cell take the KPI delta `δ_c = value_cur(c) − value_base(c)`, with
`N` cells and mean `δ̄ = (Σ δ_c)/N`.

A subset `S` of dimensions partitions the grid by the `S`-coordinates. Predict each cell by
its group mean and take the **between-group (explained) sum of squares**:

```
V(S) = Σ_c ( mean{ δ_c′ : c′ ∈ group_S(c) } − δ̄ )²
```

- `V(∅) = 0` — one group, whose mean is `δ̄`.
- `V(D) = Σ_c (δ_c − δ̄)²` — the total sum of squares; each cell is its own group.
- Adding a dimension **refines** the partition, and by the ANOVA decomposition a refinement
  never decreases the between-group sum of squares. So `V` is **monotone**.

The Shapley value of dimension `d`, over subsets `S ⊆ D \ {d}`:

```
φ_d = Σ_S  ( |S|! · (|D| − |S| − 1)! / |D|! ) · [ V(S ∪ {d}) − V(S) ]
```

Reported share: `s_d = φ_d / V(D)`.

**Exact, not sampled.** Four dimensions means 2⁴ = 16 subsets — there is no reason to
approximate. Above 8 dimensions the engine degrades to per-dimension ranking rather than
computing 2⁹ subsets.

### 3.2 The three properties, and why each matters

| Property | Statement | Consequence |
|---|---|---|
| **Efficiency** | `Σ_d φ_d = V(D) − V(∅) = V(D)` | Shares are exhaustive: nothing unattributed, nothing double-counted |
| **Null player** | a dimension that changes no coalition's value gets `φ_d = 0` | An irrelevant dimension scores exactly zero, not "a little" |
| **Symmetry** | interchangeable dimensions get equal `φ_d` | Interaction is split fairly rather than awarded to whichever is checked first |

Monotonicity plus efficiency gives `φ_d ≥ 0`, so every share is a real share.

### 3.3 Verified by hand before it touched real data

Both grids below are computed by hand in
[`test_shapley.py`](../backend/tests/test_shapley.py), which runs with no dataset and no
fixture. The function is a pure operation on a list of cells precisely so this is possible.

**Case A — one dimension carries everything.**

| δ | b1 | b2 |
|---|---|---|
| **a1** | −10 | −10 |
| **a2** | +2 | +2 |

`N=4`, `δ̄ = −4`. `V({A}) = 2(−6)² + 2(6)² = 144`; `V({B}) = 4(0)² = 0`; `V({A,B}) = 144`.

```
φ_A = ½[144 − 0] + ½[144 −   0] = 144
φ_B = ½[  0 − 0] + ½[144 − 144] =   0
Σφ = 144 = V(D)  ✓     shares: A 100%, B 0%
```

**Case B — the movement lives only in the intersection.** The planted-scenario shape.

| δ | b1 | b2 |
|---|---|---|
| **a1** | −12 | 0 |
| **a2** | 0 | 0 |

`N=4`, `δ̄ = −3`. `V({A}) = 2(−3)² + 2(3)² = 36`; `V({B}) = 36`;
`V({A,B}) = (−9)² + 3(3)² = 108`.

```
φ_A = ½[36 − 0] + ½[108 − 36] = 18 + 36 = 54
φ_B = ½[36 − 0] + ½[108 − 36] = 18 + 36 = 54
Σφ = 108 = V(D)  ✓     shares: A 50%, B 50%
```

**Case B is the entire argument for the method.** Neither dimension explains the cell alone
(36 each), but together they explain 108. The marginal contributions *within each ordering*
are unequal — 18 then 36 — because the 36 of interaction is real. Shapley splits it. Reading
the two dimensions independently gives `36 + 36 = 72` and **silently loses 36 of 108**.

That is exactly the `North` / `Product A` failure from §2.3, in miniature.

### 3.4 On the real data

```
product   φ = 1.179e10   47.4%
region    φ = 1.133e10   45.6%
segment   φ = 9.615e08    3.9%
channel   φ = 7.692e08    3.1%
```

The movement is shaped by **product and region** (93.0% between them) and essentially not at
all by channel or segment — even though `Enterprise` (a segment) carries 60.5% of the
decline and `Online` (a channel) carries 58.3%. Those two are passengers: large slices that
moved in line with their size.

---

## 4. Per-member statistical support

The over-index test cannot distinguish a genuinely disrupted segment from a small, noisy
one — a member that swings wildly every quarter clears `over_index ≥ 1.2` routinely and
means nothing by it. So each member is also tested **against its own history**, with the
same machinery `assess_significance` applies to the KPI:

1. Build the member's own quarterly series and its period-over-period % changes.
2. Take the median and **MAD-based robust sigma**, `σ = 1.4826 · median(|x − median|)`.
   MAD rather than standard deviation because the standard deviation is inflated by the very
   outlier under test.
3. Inflate for sample size: `σ ← σ · √(1 + 1/n)`.
4. `z = (current change − median) / σ`.

Fewer than 3 prior periods returns `robust_z = None` with a stated reason. It is **not**
scored as zero — see §5.3.

Implementation: [`member_significance`](../backend/app/engines/drivers.py). The robust sigma
is defined once and shared with `observe.assess_significance` so the KPI-level and
member-level tests cannot drift apart.

---

## 5. Temporal behaviour

### 5.1 Onset dating

A member's onset is the first week its weekly series leaves its own baseline band
(`median ± 1σ`, robust) **and stays out for 2 consecutive weeks** — the persistence
requirement is what stops a single noisy week being called a turning point.

`weeks_outside_band` counts breaches inside the period itself. A one-week collapse and a
quarter-long slide can produce identical contribution figures while being entirely different
problems, and only one of them is still happening.

Implementation: [`detect_onset`](../backend/app/engines/analysis.py), reused rather than
reimplemented.

### 5.2 Why onset dating is what makes a multi-factor movement legible

Three factors are planted in Q2-2026 and they start at different times. The onsets are what
separate them:

| Detected onset | Member | Factor |
|---|---|---|
| 2026-04-06 | `region = North` | demand erosion begins |
| 2026-04-20 | `product = Product A` | (already falling — North's erosion hits Product A hardest) |
| 2026-05-04 | `region = East`, `region = West` | supply disruption reaches Product A everywhere |

### 5.3 Lead/lag uses differences, never levels

Two series that are both trending correlate near-perfectly at *every* lag. Differencing
removes the shared trend and leaves the timing question, so the cross-correlation is computed
on **week-on-week changes** ([`lead_lag`](../backend/app/engines/analysis.py)).

---

## 6. The composite score

### 6.1 The formula

```
member_score = ( 0.40·M + 0.25·D + 0.20·G + 0.15·P ) / Σ(weights available)

driver_score = member_score × axis_weight(dimension)
```

| Term | Definition | Saturates at |
|---|---|---|
| `M` magnitude | `clip(|contribution_pct| / 100, 0, 1)` | 100% of the movement |
| `D` disproportion | `clip((over_index − 1.0) / 1.0, 0, 1)` | `over_index ≥ 2.0` |
| `G` significance | `clip((|z| − 1.0) / 2.0, 0, 1)` | `|z| ≥ 3.0` |
| `P` persistence | `clip(weeks_outside_band / weeks_in_period, 0, 1)` | the whole period |
| `axis_weight` | `φ_dimension / max(φ)` | leading axis = 1.0 |

### 6.2 The axis weight is the part that does the real work

Without it, ranking members across dimensions is dominated by whichever dimension cuts the
business into the fewest, largest pieces. With it, a member can only be a driver of the
*shape* of a change if its dimension explains the shape of that change — and §3 already
measures exactly that.

Concretely, on this dataset:

| Member | `member_score` | `axis_weight` | `driver_score` | Outcome |
|---|---|---|---|---|
| Enterprise | 0.581 | 0.082 | **0.048** | correctly demoted |
| Online | 0.560 | 0.065 | **0.036** | correctly demoted |
| East | 0.380 | 0.961 | **0.365** | correctly surfaced |

Ranked on member evidence alone, `Enterprise` (0.581) outranks `East` (0.380) and the third
planted factor is pushed out of the top six entirely. The axis weight is what recovers it.

### 6.3 Missing components are dropped, not zeroed

A component that cannot be computed — no history for a robust z, no weekly series for
persistence — is **removed and the remaining weights renormalised**. Scoring it as zero would
silently demote every member of a short dataset for the crime of being new. Absence of
evidence is not evidence of absence. `score_weight_used` reports how much weight was
actually available.

### 6.4 The weights are an editorial choice, and they are load-bearing only if the data is weak

The four weights are not derived from anything. They are a judgement, published as constants
in [`drivers.py`](../backend/app/engines/drivers.py) and echoed on every ranked row under
`score_components`, so a reader can recompute the ranking by hand and disagree.

The defence is empirical, not rhetorical: `scripts/validate_rca.py` re-ranks with each weight
scaled to **50% and 150%** of its published value — 8 perturbations. **All 8 leave the top
three unchanged.** If they did not, these weights would be doing the work the evidence is
supposed to do.

---

## 7. Hypothesis ranking (stages 2–4)

Driver ranking answers *where*. It does not answer *why*, and the two are deliberately kept
apart. Hypotheses are ranked by an **evidence ledger**, not by a probability:

```
confidence = support / (support + against + missing_penalty + 1.2)
```

`support` and `against` are weighted sums of measured evidence strengths. The `+1.2` prior
stops a hypothesis with one weak piece of evidence scoring highly merely because nothing has
contradicted it yet. Fixed terms:

| Signal | Effect |
|---|---|
| Cause precedes the KPI move | **+1.6** |
| KPI precedes the cause | **−2.4** |
| Cross-sectional consistency `\|r\| ≥ 0.5` | **+1.1·\|r\|** |
| Cross-sectional inconsistency | **−0.8** |
| Each counterexample member | **−0.6**, capped at −2.0 |
| Reverse causation suspected | **−1.8** |

And three hard caps, which are the point of the CONTEST stage:

- **55** when the KPI moved before the proposed cause — it cannot be the whole explanation
  however strong its evidence.
- **45** when the "cause" may be mechanically downstream of the KPI.
- **10** when there is no supporting evidence at all.

Confidence is an evidence-strength score, **not a probability**, and it does not establish
causation. Implementation: [`score_hypothesis`](../backend/app/engines/contest.py).

---

## 8. Worked example — revenue, 2026-Q2 vs 2026-Q1

Revenue fell **−17.2%** (4,433,063 → 3,671,325; Δ = −761,738). Verdict:
`meaningful_signal`.

Axis Shapley: `product 47.4% · region 45.6% · segment 3.9% · channel 3.1%`.

| # | Driver | M | D | G | P | member | × axis | **score** |
|---|---|---|---|---|---|---|---|---|
| 1 | `product = Product A` | 0.783 | 0.849 | 1.000 | 0.923 | 0.864 | 1.000 | **0.864** |
| 2 | `region = North` | 0.645 | 1.000 | 1.000 | 1.000 | 0.858 | 0.961 | **0.825** |
| 3 | `region = East` | 0.189 | 0.000 | 1.000 | 0.692 | 0.380 | 0.961 | **0.365** |
| 4 | `product = Product C` | 0.138 | 0.000 | 0.470 | 0.769 | 0.264 | 1.000 | **0.264** |
| 5 | `product = Product B` | 0.079 | 0.000 | 0.366 | 0.846 | 0.232 | 1.000 | **0.232** |
| 6 | `region = West` | 0.093 | 0.000 | 0.267 | 0.615 | 0.183 | 0.961 | **0.176** |

Checking rank 1 by hand:
`(0.40·0.7828 + 0.25·0.8487 + 0.20·1.0 + 0.15·0.9231) / 1.0 = 0.3131 + 0.2122 + 0.2000 + 0.1385 = 0.8638` ✓

Underlying figures:

| Driver | contribution | over-index | robust z | onset | weeks out |
|---|---|---|---|---|---|
| Product A | 78.3% | 1.85 | −6.72 | 2026-04-20 | 12/13 |
| North | 64.5% | 2.24 | −7.94 | 2026-04-06 | 13/13 |
| East | 18.9% | **0.93** | −4.01 | 2026-05-04 | 9/13 |

**East is the interesting case.** Its over-index is *below 1.0* — over the quarter as a whole
it contributed slightly less than its size. It is ranked third anyway, on significance
(z = −4.01, far outside its own history) and persistence (9 of 13 weeks). This is correct:
East's erosion only began on 18 May, so six weeks of severe decline are diluted by seven
normal ones in the quarter total. A quarter-granularity over-index test structurally cannot
see a factor that starts late, and had the ranking depended on over-index alone, the third
planted factor would have been missed.

---

## 9. Validation against planted ground truth

### 9.1 The scenario and how its truth is measured

`scripts/generate_sample_data.py` plants **three** factors in Q2-2026 and emits
`sample_data/ground_truth.json` alongside the CSV, so the data and its truth cannot drift.

Every factor is a toggle. `simulate()` reseeds the RNG at entry and every `jitter()` call sits
*outside* the factor conditionals, so counterfactual runs see **identical noise** and differ
only by the factor switched. All 2³ = 8 coalitions are run, giving each factor's exact
Shapley value on Q2 revenue — so the overlap between the North erosion and the Product A
supply shock (they collide in the `North × Product A` cell) is split fairly rather than
double-counted the way naive leave-one-out does. Both are recorded:

| Factor | Locus | Planted onset | Shapley | Leave-one-out | Share |
|---|---|---|---|---|---|
| Supply disruption | `product = Product A` | 2026-05-04 | −471,864 | −426,535 | 46.1% |
| North demand erosion | `region = North` | 2026-04-06 | −444,765 | −408,219 | 43.4% |
| East demand erosion | `region = East` | 2026-05-18 | −107,214 | −98,431 | 10.5% |

Σ Shapley = −1,023,842 = the total planted effect, exactly (efficiency).

### 9.2 The two denominators — an honest complication

Natural drift this quarter is **+262,105**: trend and seasonality push revenue *up* while the
planted factors push it *down*.

```
observed delta  (−761,738)  =  planted effect (−1,023,842)  +  natural drift (+262,105)
```

The planted effect is therefore **134%** of the observed movement. A member's share of the
*planted effect* and its share of the *observed movement* are different numbers, and the
manifest records both:

| Region | Planted | Planted share | Observed | Observed share | Drift |
|---|---|---|---|---|---|
| North | −568,306 | 55.5% | −491,340 | 64.5% | +76,965 |
| East | −201,980 | 19.7% | −144,306 | 18.9% | +57,674 |
| West | −135,817 | 13.3% | −71,066 | 9.3% | +64,751 |
| South | −117,740 | 11.5% | −55,025 | 7.2% | +62,715 |

The engine's `contribution_pct` is a share of the **observed** movement — so it must
reproduce the `observed share` column exactly (that is the same arithmetic, and it is checked
to 0.01pp). Attribution accuracy against the **planted** share is scored separately and
looser, because no engine should be expected to attribute natural drift to a factor.

### 9.3 The scorecard

Produced by `make validate`; asserted check-by-check in
[`test_rca_ground_truth.py`](../backend/tests/test_rca_ground_truth.py).

| Check | Result | Bar |
|---|---|---|
| **detection** | ✅ `meaningful_signal`, −17.2% | must |
| **locus recall** | ✅ **3 / 3** planted loci found | 3/3 |
| **precision @3** | ✅ **3 / 3** — Product A, North, East | ≥ 0.66 |
| **rank correlation** | ✅ **ρ = 1.00** — engine order identical to true-impact order | ≥ 0.8 |
| **decomposition exact** | ✅ worst gap **0.0005 pp** | ≤ 0.01 pp |
| **attribution MAE** | ✅ **6.4 pp** vs true planted share | ≤ 15 pp |
| **onset dating** | ✅ worst error **2 weeks** | ≤ 2 weeks |
| **temporal ordering** | ✅ `kpi_precedes_cause`, lag 4 weeks | must |
| **mechanism** | ✅ volume −783,417 vs price +21,679 → volume-led | volume-led |
| **weight sensitivity** | ✅ **8 / 8** perturbations leave the top 3 unchanged | must |

**10 / 10.** Full report: [`rca_validation.json`](rca_validation.json).

### 9.4 The finding the whole architecture exists for

Revenue in North turned the week of **2026-04-06**. The Product A fulfilment signal — the
supply disruption's own signature, and by far the best-documented explanation available in
the uploaded documents — turned **2026-05-04**, exactly its planted start date.

The decline began **four weeks before its best-evidenced cause did.** The supply disruption
is real and it is a genuine contributing factor, but it cannot explain a change that started
before it, and CONTEST caps it at 55 accordingly.

Note that this comparison must be made on the factor's **signature metric**, not on its
locus's revenue: Product A's *revenue* turns on 2026-04-20, because Product A is hit by both
factors. Dating the supply disruption by Product A's revenue would date it two weeks early
and destroy the finding.

---

## 10. Alternatives considered and rejected

**Regression coefficients over dimension dummies.** Coefficients are not a decomposition:
they do not sum to the delta, they shift under collinear dimensions (which business
dimensions always are), and they need a model specification nobody chose. Shapley over ESS
gives an exact, order-independent split with no fitted model.

**Sampled / Monte-Carlo Shapley.** Standard for hundreds of features. Here `|D| ≤ 8` means at
most 256 subsets — approximating an exactly computable quantity would add variance for
nothing, and would make the hand-verification in §3.3 impossible.

**A single "importance" number per member.** Rejected because the four components disagree
in informative ways. East ranks third on significance and persistence *despite* an over-index
below 1.0, and that disagreement is the finding. Collapsing to one number before showing the
components hides exactly the case a reader most needs to see.

**Shapley over dimension *members* rather than dimensions.** Members within a dimension are
already exactly decomposable (§2.1); Shapley there would restate arithmetic. The interaction
problem is *across* dimensions, which is where it is applied.

**A hard over-index cliff (the previous behaviour).** The ranking was
`sort by (over_index ≥ 1.2, contribution)` — a two-tier lexicographic sort. It emitted no
score, gave no reason for an ordering, and put a hard cliff between 1.19 and 1.21. It would
also have missed East entirely.

---

## 11. Limitations

**Honest about what this does not do.**

1. **No multiple-comparison correction.** Every dimension × member × KPI is scanned and the
   extreme ones reported. With enough dimensions, some member will look significant by
   chance. No FDR control is applied.
2. **The confidence weights in §7 are hand-tuned** and have never been calibrated against
   human judgement. The driver weights at least have the sensitivity sweep in §6.4; the
   confidence weights have nothing equivalent.
3. **Correlation is Pearson only** — no rank correlation, no significance test on `r`, no
   confidence interval.
4. **`price_volume_decomposition` is hard-coded to `revenue` / `units_sold`**
   ([`analysis.py`](../backend/app/engines/analysis.py)) and is not contract-aware, so it
   silently returns `None` on the hospital and school datasets.
5. **`consistency_check` is hard-coded to `("region","product","channel","segment")`**
   ([`contest.py`](../backend/app/engines/contest.py)), so cross-sectional consistency and
   counterexample search return `not_applicable` on datasets with other dimension names —
   which also removes the `±1.1·|r|` and `−0.6·n` confidence terms on those datasets.
6. **The Shapley axis weighting assumes dimensions are roughly balanced.** A dimension with
   one member per row (an ID column mistaken for a dimension) would explain everything
   trivially. Schema detection caps dimension cardinality, but the interaction is unguarded.
7. **Ground truth is simulated, not observed.** It proves the engine recovers factors that
   were planted by a known generative process. It does not prove behaviour on real business
   data, where causes are not toggles and noise is not Gaussian.
8. **One scenario, one KPI, one period.** The scorecard validates revenue in 2026-Q2. It is
   not a benchmark suite.

---

## 12. Where the code lives

| Concern | File |
|---|---|
| Shapley attribution, member significance, persistence, composite score | [`backend/app/engines/drivers.py`](../backend/app/engines/drivers.py) |
| Contribution + rate/mix decomposition, KPI significance | [`backend/app/engines/observe.py`](../backend/app/engines/observe.py) |
| Onset dating, lead/lag, correlation, counterexamples, price/volume | [`backend/app/engines/analysis.py`](../backend/app/engines/analysis.py) |
| Hypothesis confidence ledger and caps | [`backend/app/engines/contest.py`](../backend/app/engines/contest.py) |
| Scenario generator + counterfactual ground truth | [`scripts/generate_sample_data.py`](../scripts/generate_sample_data.py) |
| Scorecard harness | [`scripts/validate_rca.py`](../scripts/validate_rca.py) |
| Shapley axioms, hand-computed grids | [`backend/tests/test_shapley.py`](../backend/tests/test_shapley.py) |
| Ground-truth recovery | [`backend/tests/test_rca_ground_truth.py`](../backend/tests/test_rca_ground_truth.py) |
