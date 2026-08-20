# Demo video script

**Target length: 3 minutes.** Record at 1920×1080. Use the Data Analyst role so the
statistical layer is visible.

## Before recording

```bash
# terminal 1
cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8000
# terminal 2
cd frontend && npm run dev
```

In the app: sign up as **Data Analyst** → **Business data → Load sample company** →
**Documents → Load sample documents** → **Dashboard → 2026 / Q2**. Then reload and start
recording from the sign-in page.

---

## 0:00 – 0:20 · The problem

> "Every business dashboard can tell you revenue fell 17% last quarter. Almost none of them
> can tell you whether that matters, why it happened, or whether the explanation you have been
> given is actually true. That last part is where this project spends its effort."

*Sign in. Land on the dashboard.*

## 0:20 – 0:50 · Observe — is it even real?

*Point at the headline: −17.2%, "Meaningful signal".*

> "First question: is this a real signal or normal noise? The system compares this quarter
> against the same quarter transition in previous years, using robust statistics, and finds it
> is 9.2 standard deviations outside what this business normally does. The shaded band on the
> chart is that normal range."

*Expand "Analyst detail — significance method".*

> "The analyst can see exactly how: the method, the z-score, the historical distribution — and
> a note explaining that the same-quarter sample was small, so the dispersion estimate was
> floored. The system tells you when its own test is weak."

*Point at the drivers.*

> "Then: which part of the business? Not the biggest part — the disproportionate part. North
> is 29% of the business and 65% of the decline: 2.2 times its own size. Enterprise is 60% of
> the decline, but it is also 60% of the business, so it is arithmetic, not a driver."

## 0:50 – 1:20 · Investigate — competing explanations

*Investigation → Run investigation → Investigate tab.*

> "Now: what could explain it? Not one explanation — five competing ones, drawn from ten the
> system considered and filtered to those this dataset can actually test. It will not propose
> an explanation it cannot examine."

> "Each is tested against two kinds of evidence: numbers computed from the uploaded rows, and
> passages retrieved from the uploaded documents — quoted verbatim, with the file and section
> shown."

## 1:20 – 2:10 · Contest — the part that matters

*Contest tab. Open the supply hypothesis.*

> "Here is the interesting part. The operations report documents a supply disruption in
> detail — inventory collapse, stockouts, allocation, dates. It is by far the best-evidenced
> explanation available, and it is the one a language model would confidently pick."

*Scroll to the "Which moved first?" chart.*

> "So the system dates both series. Revenue in North turned on the 6th of April. Stockouts
> turned on the 4th of May — four weeks later. **A cause cannot explain a change that started
> before it.**"

> "Its confidence is capped at 55%, the reason is shown, and it is labelled a contributing
> factor rather than the root cause. A less well-documented explanation — competitive demand
> erosion, which *did* start first — ranks above it."

*Point at the marketing hypothesis.*

> "And this one is flagged for reverse causation: marketing spend is budgeted as a share of
> revenue, so it falls *because* revenue fell. It correlates almost perfectly and explains
> nothing. The system knows the difference."

## 2:10 – 2:40 · Act — what to do

*Act tab.*

> "Finally: what to do. Each recommendation names the hypothesis it rests on and that
> hypothesis's confidence. It carries the evidence behind it. It sets an early-warning
> threshold computed from this business's own weekly history — the operations report noted
> there was no stockout alerting, and this is the concrete fix."

> "And each one states what would change the advice. The uncertainty is carried through to the
> leader, not hidden from them."

## 2:40 – 3:00 · The architecture claim

*Settings page, or the architecture diagram.*

> "Nothing you just saw was a number written by a language model. Every figure is computed by
> a deterministic pandas layer from the uploaded data; every quotation is retrieved from an
> uploaded document. The model frames, classifies and writes — over facts it is given. In fact
> the whole four-stage pipeline runs with no API key at all."

> "Business intelligence should not stop at telling us what happened. It should help us
> understand why, challenge that explanation, and decide what to do next."

---

## Points worth landing

1. Meaningful signal vs normal variation — the system refuses to treat every wobble as news.
2. Over-indexing — the biggest segment is not automatically the driver.
3. Competing hypotheses, not one confident story.
4. **The temporal contradiction** — the strongest single moment in the demo.
5. Reverse-causation screen — correlation is not causation, enforced in code.
6. Uncertainty preserved into the recommendation.
7. The LLM never produces a number.

## If something breaks on the day

* Open `docs/ui-preview.html` — a self-contained page rendered from a real run, no servers.
* Or run `python3 scripts/run_pipeline_demo.py` and narrate the terminal output.
