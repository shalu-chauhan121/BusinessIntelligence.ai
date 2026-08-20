"""
Build a self-contained HTML preview of the product UI from a real pipeline run.

    python3 scripts/run_pipeline_demo.py          # produces demo_output/investigation.json
    python3 scripts/build_ui_preview.py           # produces docs/ui-preview.html

The preview uses the same design tokens, layout and chart specifications as the
React application, and is rendered entirely from the JSON the backend actually
returned — so it is an honest picture of the product, not a mock-up. It needs no
Node, no network and no backend, which makes it a safe fallback for a demo.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "demo_output" / "investigation.json"
OUT = ROOT / "docs" / "ui-preview.html"

TOKENS = """
:root,:root[data-theme='light']{color-scheme:light;--surface-1:#fcfcfb;--plane:#f4f4f1;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#898781;--gridline:#e1e0d9;--baseline:#c3c2b7;--border:rgba(11,11,11,.10);
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--diverge-pos:#2a78d6;--diverge-neg:#e34948;
--status-good:#0ca30c;--status-warning:#fab219;--status-serious:#ec835a;--status-critical:#d03b3b;
--delta-good:#006300;--delta-bad:#c02727;--band-fill:rgba(42,120,214,.10)}
:root[data-theme='dark']{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;
--text-secondary:#c3c2b7;--text-muted:#898781;--gridline:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--diverge-pos:#3987e5;--diverge-neg:#e66767;
--status-good:#0ca30c;--status-warning:#fab219;--status-serious:#ec835a;--status-critical:#d03b3b;
--delta-good:#0ca30c;--delta-bad:#e66767;--band-fill:rgba(57,135,229,.12)}
"""

CSS = TOKENS + """
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--text-primary);
font-family:system-ui,-apple-system,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased;font-size:14px;line-height:1.5}
.tnum{font-variant-numeric:tabular-nums}
header.top{position:sticky;top:0;z-index:5;border-bottom:1px solid var(--border);background:var(--surface-1)}
.top-inner,.wrap{max-width:1240px;margin:0 auto;padding:0 20px}
.top-inner{height:56px;display:flex;align-items:center;gap:10px}
.logo{width:28px;height:28px;border-radius:6px;background:var(--series-1);display:grid;place-items:center;color:#fff;font-weight:700;font-size:13px}
.brand{font-weight:600;letter-spacing:-.01em}
.brand span{color:var(--series-1)}
.spacer{margin-left:auto}
.layout{max-width:1240px;margin:0 auto;padding:18px 20px 64px;display:grid;grid-template-columns:200px 1fr;gap:22px}
nav.side ul{list-style:none;margin:0;padding:0;position:sticky;top:76px}
nav.side a{display:flex;gap:9px;align-items:center;padding:8px 11px;border-radius:8px;color:var(--text-secondary);
text-decoration:none;font-size:13.5px}
nav.side a.on{background:var(--plane);color:var(--text-primary);font-weight:600}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px}
.pad{padding:18px}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
h1,h2,h3,h4{margin:0;letter-spacing:-.01em}
h2{font-size:17px;font-weight:600}
h3{font-size:15px;font-weight:600}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--text-muted)}
.muted{color:var(--text-muted)}
.sec{color:var(--text-secondary)}
.sub{color:var(--text-secondary);font-size:13px;margin-top:4px;max-width:70ch}
.row{display:flex;align-items:center;gap:10px}
.between{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:3px 10px;font-size:11.5px;font-weight:600}
.hero{font-size:34px;font-weight:600;letter-spacing:-.02em}
.delta{font-size:17px;font-weight:600}
.kpi{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px}
.kpi .lab{font-size:12px;color:var(--text-secondary)}
.kpi .val{font-size:21px;font-weight:600;margin-top:5px}
.kpi .chg{font-size:12px;margin-top:3px;font-weight:600}
.stages{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}
.stage{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:13px;cursor:default}
.stage.on{outline:2px solid var(--series-1);outline-offset:-1px}
.stage .n{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted)}
.stage .t{font-weight:600;margin-top:3px}
.stage .q{font-size:12.5px;color:var(--text-secondary)}
.hyp{border:1px solid var(--border);border-radius:10px;background:var(--surface-1);margin-bottom:12px;overflow:hidden}
.hyp .head{display:flex;gap:12px;padding:16px 18px;align-items:flex-start}
.rank{width:24px;height:24px;border-radius:999px;background:var(--plane);color:var(--text-secondary);
display:grid;place-items:center;font-size:12px;font-weight:700;flex:0 0 auto}
.meter{width:132px;flex:0 0 auto}
.meter .bar{height:6px;border-radius:999px;background:var(--gridline);overflow:hidden;margin-top:4px}
.meter .fill{height:100%;border-radius:999px}
.body{border-top:1px solid var(--border);padding:16px 18px}
.ev{border:1px solid var(--border);border-radius:8px;padding:9px 11px;margin-bottom:8px}
.ev .t{font-weight:600;font-size:13px}
.ev .d{font-size:13px;color:var(--text-secondary);margin-top:2px}
.quote{border-left:3px solid var(--status-good);border:1px solid var(--border);border-left-width:3px;border-radius:8px;
padding:9px 11px;margin-bottom:8px}
.quote.neg{border-left-color:var(--status-critical)}
.quote p{margin:0;font-style:italic;font-size:13px;color:var(--text-secondary)}
.quote .src{font-size:11.5px;color:var(--text-muted);margin-top:6px}
.callout{border-left:4px solid var(--series-1);background:var(--plane);border-radius:8px;padding:10px 12px;font-size:13px}
.callout.warn{border-left-color:var(--status-warning)}
.callout.bad{border-left-color:var(--status-critical)}
.callout b{display:block;color:var(--text-primary)}
.callout div{color:var(--text-secondary)}
ul.clean{list-style:none;margin:8px 0 0;padding:0}
ul.clean li{font-size:13px;color:var(--text-secondary);margin-bottom:5px;padding-left:14px;position:relative}
ul.clean li:before{content:'';position:absolute;left:2px;top:8px;width:4px;height:4px;border-radius:999px;background:var(--baseline)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--text-muted);font-weight:600;padding:4px 10px 4px 0}
td{padding:4px 10px 4px 0;color:var(--text-secondary)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--text-secondary)}
.swatch{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}
.driver{display:flex;justify-content:space-between;align-items:center;gap:12px;border:1px solid var(--border);
border-radius:8px;padding:9px 11px;margin-bottom:7px}
.sel{border:1px solid var(--border);background:var(--surface-1);border-radius:8px;padding:7px 10px;font-size:13px;color:var(--text-primary);width:100%}
.lbl{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted);margin-bottom:5px;display:block}
.banner{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:12.5px;
color:var(--text-secondary);margin-bottom:14px}
@media (max-width:900px){.layout{grid-template-columns:1fr}nav.side{display:none}.g2,.g3,.g4,.stages{grid-template-columns:1fr}}
"""


# ---------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def fmt(value, unit="count", compact=True):
    if value is None:
        return "—"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "ratio":
        return f"{value:,.2f}"
    a = abs(value)
    if compact and a >= 1e9:
        return f"{value/1e9:.2f}B"
    if compact and a >= 1e6:
        return f"{value/1e6:.2f}M"
    if compact and a >= 1e3:
        return f"{value/1e3:.1f}K"
    return f"{value:,.0f}"


def delta(v):
    return "—" if v is None else f"{v:+.1f}%"


BAND = {
    "strong": ("Strong evidence", "var(--status-good)"),
    "moderate": ("Moderate evidence", "var(--status-warning)"),
    "weak": ("Weak evidence", "var(--status-serious)"),
    "insufficient": ("Insufficient evidence", "var(--status-critical)"),
}


def chip(text, color, bg):
    return f'<span class="chip" style="color:{color};background:{bg}">{esc(text)}</span>'


def tone_chip(text, tone):
    palette = {
        "good": ("var(--status-good)", "rgba(12,163,12,.12)"),
        "warning": ("var(--status-warning)", "rgba(250,178,25,.16)"),
        "serious": ("var(--status-serious)", "rgba(236,131,90,.16)"),
        "critical": ("var(--status-critical)", "rgba(208,59,59,.14)"),
        "neutral": ("var(--text-secondary)", "var(--plane)"),
    }
    c, b = palette.get(tone, palette["neutral"])
    return chip(text, c, b)


# ---------------------------------------------------------------------------
# charts (inline SVG, same specs as the React components)
# ---------------------------------------------------------------------------
def trend_svg(obs, w=560, h=230):
    series = [s for s in obs["series"]["quarterly"] if s["value"] is not None]
    if len(series) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 52, 14, 14, 26
    values = [s["value"] for s in series]
    rng = obs["significance"].get("normal_range")
    lo_v, hi_v = min(values), max(values)
    if rng:
        lo_v, hi_v = min(lo_v, rng[0]), max(hi_v, rng[1])
    span = (hi_v - lo_v) or 1
    lo_v -= span * 0.12
    hi_v += span * 0.12
    span = hi_v - lo_v

    def x(i):
        return pad_l + i * (w - pad_l - pad_r) / max(len(series) - 1, 1)

    def y(v):
        return pad_t + (hi_v - v) / span * (h - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" '
             f'aria-label="{esc(obs["kpi_label"])} by quarter">']
    for frac in (0, .25, .5, .75, 1):
        gy = pad_t + frac * (h - pad_t - pad_b)
        val = hi_v - frac * span
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w-pad_r}" y2="{gy:.1f}" stroke="var(--gridline)"/>')
        parts.append(f'<text x="{pad_l-8}" y="{gy+3.5:.1f}" text-anchor="end" font-size="10" '
                     f'fill="var(--text-muted)" class="tnum">{fmt(val, obs["unit"])}</text>')
    if rng:
        parts.append(f'<rect x="{pad_l}" y="{y(rng[1]):.1f}" width="{w-pad_l-pad_r}" '
                     f'height="{max(y(rng[0])-y(rng[1]),1):.1f}" fill="var(--band-fill)"/>')
    exp = obs["significance"].get("expected_value")
    if exp:
        parts.append(f'<line x1="{pad_l}" y1="{y(exp):.1f}" x2="{w-pad_r}" y2="{y(exp):.1f}" '
                     f'stroke="var(--text-muted)" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{pad_l+6}" y="{y(exp)-6:.1f}" font-size="10" fill="var(--text-muted)">expected</text>')
    d = " ".join(f'{"M" if i == 0 else "L"}{x(i):.1f},{y(s["value"]):.1f}' for i, s in enumerate(series))
    parts.append(f'<path d="{d}" fill="none" stroke="var(--series-1)" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    cur = obs["timeframe"]["label"]
    for i, s in enumerate(series):
        if s["period"] == cur:
            colour = "var(--status-critical)" if obs["anomaly"] else "var(--series-1)"
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(s["value"]):.1f}" r="5.5" fill="{colour}" '
                         f'stroke="var(--surface-1)" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(s["value"]):.1f}" r="2.4" fill="var(--series-1)"/>')
    step = max(1, len(series) // 7)
    for i, s in enumerate(series):
        if i % step == 0 or s["period"] == cur:
            parts.append(f'<text x="{x(i):.1f}" y="{h-8}" text-anchor="middle" font-size="10" '
                         f'fill="var(--text-muted)">{esc(s["period"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def driver_svg(rows, unit, w=560, bar=26):
    rows = [r for r in rows if r.get("change_abs") is not None and not r.get("is_aggregate")]
    if not rows:
        return ""
    h = len(rows) * bar + 26
    label_w, pad_r = 116, 54
    plot = w - label_w - pad_r
    limit = max(abs(r["change_abs"]) for r in rows) or 1
    zero = label_w + plot / 2

    def px(v):
        return v / limit * (plot / 2)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Contribution by driver">']
    parts.append(f'<line x1="{zero}" y1="4" x2="{zero}" y2="{h-22}" stroke="var(--baseline)"/>')
    for i, r in enumerate(rows):
        cy = 6 + i * bar
        val = r["change_abs"]
        colour = "var(--diverge-neg)" if val < 0 else "var(--diverge-pos)"
        length = abs(px(val))
        x0 = zero - length if val < 0 else zero
        parts.append(f'<rect x="{x0:.1f}" y="{cy:.1f}" width="{max(length,1):.1f}" height="{bar-10}" '
                     f'rx="4" fill="{colour}"/>')
        parts.append(f'<text x="{label_w-10}" y="{cy+(bar-10)/2+3.5:.1f}" text-anchor="end" font-size="11.5" '
                     f'fill="var(--text-secondary)">{esc(r["name"])[:18]}</text>')
        contrib = r.get("contribution_pct")
        if contrib is not None:
            parts.append(f'<text x="{w-pad_r+8}" y="{cy+(bar-10)/2+3.5:.1f}" font-size="11" '
                         f'fill="var(--text-secondary)" class="tnum">{contrib:.0f}%</text>')
    parts.append(f'<text x="{zero}" y="{h-6}" text-anchor="middle" font-size="10" fill="var(--text-muted)">0</text>')
    parts.append("</svg>")
    return "".join(parts)


def onset_svg(kpi_series, cause_series, kpi_onset, cause_onset, kpi_label, cause_label, w=560, h=220):
    if not kpi_series or not cause_series:
        return ""

    def standardise(rows):
        """Deviation from the series' own pre-period normal, in robust sigmas."""
        vals = [r["value"] for r in rows if r["value"] is not None]
        window = vals[: max(6, len(vals) // 3)]
        if len(window) < 3:
            return [(r["week"], None) for r in rows]
        med = sorted(window)[len(window) // 2]
        devs = sorted(abs(v - med) for v in window)
        sigma = 1.4826 * devs[len(devs) // 2]
        if not sigma:
            mean = sum(window) / len(window)
            sigma = (sum((v - mean) ** 2 for v in window) / max(len(window) - 1, 1)) ** 0.5
        if not sigma:
            return [(r["week"], None) for r in rows]
        return [(r["week"], (r["value"] - med) / sigma if r["value"] is not None else None) for r in rows]

    a, b = standardise(kpi_series), standardise(cause_series)
    bmap = dict(b)
    weeks = [w0 for w0, _ in a]
    pad_l, pad_r, pad_t, pad_b = 44, 12, 18, 26
    pts = [v for _, v in a if v is not None] + [v for v in bmap.values() if v is not None]
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1
    lo -= span * .1
    hi += span * .1
    span = hi - lo

    def x(i):
        return pad_l + i * (w - pad_l - pad_r) / max(len(weeks) - 1, 1)

    def y(v):
        return pad_t + (hi - v) / span * (h - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Which moved first">']
    for frac in (0, .5, 1):
        gy = pad_t + frac * (h - pad_t - pad_b)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w-pad_r}" y2="{gy:.1f}" stroke="var(--gridline)"/>')
        parts.append(f'<text x="{pad_l-7}" y="{gy+3.5:.1f}" text-anchor="end" font-size="10" '
                     f'fill="var(--text-muted)" class="tnum">{hi - frac*span:.0f}&#963;</text>')

    def path(pairs, colour):
        d, started = [], False
        for i, (_, v) in enumerate(pairs):
            if v is None:
                continue
            d.append(f'{"M" if not started else "L"}{x(i):.1f},{y(v):.1f}')
            started = True
        return f'<path d="{" ".join(d)}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>'

    parts.append(path(a, "var(--series-1)"))
    parts.append(path([(wk, bmap.get(wk)) for wk in weeks], "var(--series-2)"))
    for idx, (week, colour, text) in enumerate(((kpi_onset, "var(--series-1)", "KPI turns"),
                                                (cause_onset, "var(--series-2)", "cause turns"))):
        if week in weeks:
            xi = x(weeks.index(week))
            label_y = pad_t + 9 + idx * 13          # stagger so coincident onsets stay legible
            parts.append(f'<line x1="{xi:.1f}" y1="{pad_t}" x2="{xi:.1f}" y2="{h-pad_b}" stroke="{colour}" '
                         f'stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{xi+4:.1f}" y="{label_y}" font-size="10" fill="{colour}">{text}</text>')
    step = max(1, len(weeks) // 6)
    for i, wk in enumerate(weeks):
        if i % step == 0:
            parts.append(f'<text x="{x(i):.1f}" y="{h-8}" text-anchor="middle" font-size="9.5" '
                         f'fill="var(--text-muted)">{esc(wk[5:])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
def build(data: dict) -> str:
    obs, inv, con, act = data["observe"], data["investigate"], data["contest"], data["act"]
    tf, base = obs["timeframe"], obs["baseline_timeframe"]

    verdict_map = {
        "meaningful_signal": ("Meaningful signal", "critical"),
        "within_normal_variation": ("Within normal variation", "good"),
        "statistically_unusual_but_immaterial": ("Unusual but immaterial", "warning"),
    }
    vlabel, vtone = verdict_map.get(obs["verdict"], ("—", "neutral"))
    years = sorted({s["year"] for s in obs["series"]["quarterly"]}, reverse=True)

    kpis = "".join(
        f'<div class="kpi"><div class="lab">{esc(k["label"])}'
        + ('<span style="float:right;font-size:10px;font-weight:700;letter-spacing:.08em;color:var(--series-1)">PRIMARY</span>' if k["is_primary"] else "")
        + f'</div><div class="val tnum">{fmt(k["current"], k["unit"])}</div>'
        f'<div class="chg tnum" style="color:{"var(--delta-good)" if ((k["change_pct"] or 0) > 0) == k["higher_is_better"] else "var(--delta-bad)"}">'
        f'{delta(k["change_pct"])} <span class="muted" style="font-weight:400">vs baseline</span></div></div>'
        for k in obs["kpi_scoreboard"][:8]
    )

    drivers = "".join(
        f'<div class="driver"><div><div style="font-weight:600;font-size:13.5px">{esc(d["name"])} '
        + (tone_chip(f'{d["over_index"]:.1f}x its size', "critical") if d.get("is_disproportionate")
           else '<span class="muted" style="font-size:11.5px">in line with its size</span>')
        + f'</div><div class="muted" style="font-size:12px;text-transform:capitalize">{esc(d["dimension"])}</div></div>'
        f'<div style="text-align:right"><div class="tnum" style="font-weight:600">{d["contribution_pct"]:.0f}%</div>'
        f'<div class="tnum" style="font-size:12px;color:var(--delta-bad)">{delta(d["change_pct"])}</div></div></div>'
        for d in obs["top_drivers"][:5]
    )

    first_dim = next(iter(obs["drivers"])) if obs.get("drivers") else None
    driver_chart = driver_svg(obs["drivers"][first_dim], obs["unit"]) if first_dim else ""

    # hypothesis cards
    cards = []
    for i, h in enumerate(con["hypotheses"]):
        sc, ct = h["scoring"], h["contest"]
        band_label, band_colour = BAND.get(sc["confidence_band"], BAND["insufficient"])
        temporal, mech, cons = ct["temporal"], ct.get("mechanism", {}), ct["consistency"]
        flags = ""
        if temporal.get("status") == "kpi_precedes_cause":
            flags += tone_chip("Started too late", "critical")
        if mech.get("reverse_causation_suspected"):
            flags += tone_chip("Reverse causation risk", "serious")
        if sc.get("causally_consistent"):
            flags += tone_chip("Timing holds", "good")

        supporting = "".join(
            f'<div class="ev"><div class="t">{esc(e["label"])} '
            f'<span style="font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:'
            f'{"var(--status-good)" if e["stance"]=="supporting" else "var(--text-muted)"}">'
            f'{"supports" if e["stance"]=="supporting" else "context"}</span></div>'
            f'<div class="d">{esc(e["detail"])}</div></div>'
            for e in h.get("evidence", []) if e["stance"] != "contradicting"
        ) or '<div class="sec" style="font-size:13px">No structured test available.</div>'

        docs = "".join(
            f'<div class="quote"><p>&ldquo;{esc(d["quote"][:240])}&rdquo;</p>'
            f'<div class="src">{esc(d["source"])}{" &sect; " + esc(d["section"]) if d.get("section") else ""}</div></div>'
            for d in h.get("documentary_evidence", [])[:2]
        ) or '<div class="sec" style="font-size:13px">Nothing in the uploaded documents supports this.</div>'

        contra = "".join(
            f'<div class="quote neg"><p>&ldquo;{esc(d["quote"][:240])}&rdquo;</p>'
            f'<div class="src">{esc(d["source"])}</div></div>'
            for d in ct.get("contradictory_evidence", [])[:2]
        ) or '<div class="sec" style="font-size:13px">No passage argues against this.</div>'

        missing = "".join(f'<li>{esc(m)}</li>' for m in h.get("missing", [])[:3])

        onset = ""
        if i == 0 or temporal.get("status") == "kpi_precedes_cause":
            onset = onset_svg(
                ct.get("temporal_series", {}).get("kpi", []),
                ct.get("temporal_series", {}).get("cause", []),
                temporal.get("kpi_onset_week"), temporal.get("cause_onset_week"),
                obs["kpi_label"], temporal.get("cause_metric_label", "cause"),
            )
            if onset:
                onset = (
                    '<div style="margin-top:16px"><h4 style="font-size:13px">Which moved first?</h4>'
                    '<div class="sub" style="margin-bottom:6px">Both series shown as deviations from their own pre-period normal, '
                    'in robust standard deviations, so one axis carries both.</div>'
                    f'<div class="legend" style="margin-bottom:6px">'
                    f'<span><i class="swatch" style="background:var(--series-1)"></i>{esc(obs["kpi_label"])}</span>'
                    f'<span><i class="swatch" style="background:var(--series-2)"></i>'
                    f'{esc(temporal.get("cause_metric_label","Proposed cause"))}</span></div>' + onset + "</div>"
                )

        cap = (f'<div class="callout bad" style="margin-bottom:12px"><b>Confidence was capped</b>'
               f'<div>{esc(sc["cap_reason"])}</div></div>') if sc.get("cap_reason") else ""

        cards.append(f"""
<article class="hyp">
  <div class="head">
    <span class="rank tnum">{i+1}</span>
    <div style="flex:1;min-width:0">
      <div class="row" style="flex-wrap:wrap">
        <h3>{esc(h['title'])}</h3>{flags}
      </div>
      <div class="sub">{esc(h['statement'])}</div>
    </div>
    <div class="meter">
      <div class="row" style="justify-content:space-between"><span class="tnum" style="font-weight:600">{sc['confidence']}%</span></div>
      <div class="bar"><div class="fill" style="width:{sc['confidence']}%;background:{band_colour}"></div></div>
      <div class="muted" style="font-size:11px;margin-top:4px">{band_label}</div>
    </div>
  </div>
  <div class="body">
    {cap}
    <div class="callout" style="margin-bottom:14px"><b>What this confidence means</b><div>{esc(sc['causal_claim'])}</div></div>
    <div class="grid g2">
      <div>
        <div class="eyebrow" style="margin-bottom:7px">Supporting evidence — measured</div>
        {supporting}
        <div class="eyebrow" style="margin:14px 0 7px">Supporting evidence — documents</div>
        {docs}
      </div>
      <div>
        <div class="eyebrow" style="margin-bottom:7px">Contradictory evidence</div>
        <div class="ev"><div class="t">Temporal precedence</div><div class="d">{esc(temporal.get('detail',''))}</div></div>
        <div class="ev"><div class="t">Consistency across the business</div>
          <div class="d">{esc((cons.get('correlation') or {}).get('interpretation') or cons.get('detail',''))}</div></div>
        {f'<div class="ev"><div class="t">Reverse-causation screen</div><div class="d">{esc(mech.get("conclusion",""))}</div></div>' if mech.get('declared_risk') else ''}
        {contra}
        <div class="eyebrow" style="margin:14px 0 7px">Missing evidence</div>
        <ul class="clean">{missing}</ul>
      </div>
    </div>
    {onset}
  </div>
</article>""")

    recs = "".join(f"""
<article class="card pad" style="margin-bottom:12px">
  <div class="between">
    <h3>{esc(r['title'])}</h3>
    <div class="row">{tone_chip(r['priority'] + ' priority', 'critical' if r['priority']=='high' else 'warning' if r['priority']=='medium' else 'neutral')}{tone_chip(r['horizon'],'neutral')}</div>
  </div>
  <div class="sub">{esc(r['rationale'])}</div>
  <ul class="clean">{''.join(f'<li>{esc(a)}</li>' for a in r['actions'])}</ul>
  <div class="muted" style="font-size:12px;margin-top:10px">Owner: {esc(r['owner'])} · Based on: {esc(r['based_on']['hypothesis'])} ({r['based_on']['confidence']}% {esc(r['based_on']['band'])})</div>
  {('<div style="margin-top:12px"><div class="eyebrow">Early-warning threshold, from your own history</div><ul class="clean">' + ''.join(f'<li>{esc(m["rule"])}</li>' for m in r['monitoring'][:2]) + '</ul></div>') if r.get('monitoring') else ''}
  <div style="margin-top:12px"><div class="eyebrow">What would change this advice</div>
    <ul class="clean">{''.join(f'<li>{esc(w)}</li>' for w in r.get('what_would_change_this', [])[:2])}</ul></div>
</article>""" for r in act["recommendations"])

    ranking = "".join(
        f'<tr><td class="tnum" style="width:26px">{r["rank"]}</td><td style="color:var(--text-primary)">{esc(r["title"])}</td>'
        f'<td class="tnum" style="text-align:right;width:60px;font-weight:600;color:var(--text-primary)">{r["confidence"]}%</td>'
        f'<td style="width:120px">{BAND.get(r["band"], BAND["insufficient"])[0]}</td></tr>'
        for r in con["ranking"]
    )

    nav_items = [("Dashboard", True), ("Investigation", False), ("History", False),
                 ("Business data", False), ("Documents", False), ("Settings", False)]
    nav = "".join(f'<li><a href="#" class="{"on" if on else ""}">{esc(n)}</a></li>' for n, on in nav_items)

    hyp_summary = "".join(
        f'<article class="card pad" style="margin-bottom:10px"><div class="between">'
        f'<h3>{i+1}. {esc(h["title"])}</h3>{tone_chip(h["family"].replace("_"," ").title(),"neutral")}</div>'
        f'<div class="sub">{esc(h["statement"])}</div>'
        f'<div class="grid g3" style="margin-top:11px">'
        f'<div style="background:var(--plane);border-radius:8px;padding:8px 10px"><div class="muted" style="font-size:11.5px">Structured tests</div><div class="tnum" style="font-weight:600">{len(h.get("evidence",[]))}</div></div>'
        f'<div style="background:var(--plane);border-radius:8px;padding:8px 10px"><div class="muted" style="font-size:11.5px">Document passages</div><div class="tnum" style="font-weight:600">{len(h.get("documentary_evidence",[]))}</div></div>'
        f'<div style="background:var(--plane);border-radius:8px;padding:8px 10px"><div class="muted" style="font-size:11.5px">Known gaps</div><div class="tnum" style="font-weight:600">{len(h.get("missing",[]))}</div></div>'
        f'</div></article>'
        for i, h in enumerate(inv["hypotheses"])
    )

    focus_chips = "".join(tone_chip(f"Focus: {m} ({d})", "critical") for d, m in (inv.get("focus") or {}).items())
    n = act["narrative"]

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BusinessIntelligence.ai — UI preview</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><div class="top-inner">
  <span class="logo">BI</span><span class="brand">BusinessIntelligence<span>.ai</span></span>
  <span class="spacer"></span>
  <button onclick="document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'"
          style="background:none;border:1px solid var(--border);border-radius:8px;padding:6px 11px;color:var(--text-secondary);cursor:pointer;font-size:12.5px">
    Toggle theme</button>
  <div style="text-align:right;line-height:1.25;margin-left:12px">
    <div style="font-size:12px;font-weight:600">Demo Analyst</div>
    <div class="muted" style="font-size:11px">Data Analyst</div></div>
</div></header>

<div class="layout">
<nav class="side"><ul>{nav}</ul></nav>
<main>
  <div class="banner">Static preview rendered from a real pipeline run
    (<code>demo_output/investigation.json</code>). Every figure below was computed by the backend&rsquo;s
    analysis layer from the sample dataset — nothing here is hand-written.</div>

  <div class="between" style="margin-bottom:14px">
    <div><div class="eyebrow">Stage 1 — Observe</div><h2>What actually changed?</h2>
      <div class="sub">{esc(data['dataset']['filename'])} · {data['dataset']['rows']:,} rows · weekly grain</div></div>
  </div>

  <div class="card pad" style="margin-bottom:14px">
    <div class="eyebrow" style="margin-bottom:10px">Analysis period</div>
    <div class="grid g4">
      <div><label class="lbl">Year</label><select class="sel">{''.join(f'<option {"selected" if y==tf["year"] else ""}>{y}</option>' for y in years)}</select></div>
      <div><label class="lbl">Quarter</label><select class="sel">{''.join(f'<option {"selected" if q==tf["quarter"] else ""}>Q{q}</option>' for q in (1,2,3,4))}<option>Full year</option></select></div>
      <div><label class="lbl">KPI</label><select class="sel">{''.join(f'<option {"selected" if k["key"]==obs["kpi"] else ""}>{esc(k["label"])}</option>' for k in obs["kpi_scoreboard"])}</select></div>
      <div><label class="lbl">Compare with</label><select class="sel"><option>Previous quarter</option><option>Same quarter last year</option></select></div>
    </div>
    <div class="muted" style="font-size:12px;margin-top:11px">Every figure on this page is recomputed from your uploaded data for the selected period.</div>
  </div>

  <div class="card pad" style="margin-bottom:14px">
    <div class="between">
      <div>
        <div class="eyebrow">{esc(obs['kpi_label'])} · {esc(tf['pretty'])} vs {esc(base['pretty'])}</div>
        <div class="row" style="align-items:baseline;margin-top:6px">
          <span class="hero tnum">{fmt(obs['current_value'], obs['unit'])}</span>
          <span class="delta tnum" style="color:{'var(--delta-bad)' if obs['is_unfavourable'] else 'var(--delta-good)'}">{delta(obs['change_pct'])}</span>
        </div>
        <div class="sub">This change is outside the range this measure normally moves in.
          Typically this comparison moves by about {delta(obs['significance']['median_historical_change_pct'])}.</div>
      </div>
      {tone_chip(vlabel, vtone)}
    </div>
    <details style="margin-top:14px;border:1px solid var(--border);border-radius:8px">
      <summary style="cursor:pointer;padding:8px 12px;font-size:11.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--text-muted)">
        Analyst detail — significance method</summary>
      <div style="border-top:1px solid var(--border);padding:12px">
        <table><tbody>
          <tr><th>Method</th><td class="tnum">{esc(obs['significance']['method'])}</td>
              <th>Robust z-score</th><td class="tnum">{obs['significance']['robust_z']:.2f}</td></tr>
          <tr><th>Median historical change</th><td class="tnum">{delta(obs['significance']['median_historical_change_pct'])}</td>
              <th>Robust sigma</th><td class="tnum">{delta(obs['significance']['robust_sigma_pct'])}</td></tr>
          <tr><th>History points</th><td class="tnum">{obs['significance']['history_points']}</td>
              <th>Same-quarter points</th><td class="tnum">{obs['significance']['same_quarter_points']}</td></tr>
        </tbody></table>
        <div class="sub">{esc(obs['significance']['statistical_power'])}</div>
        <div class="sub">{esc(obs['significance'].get('dispersion_note') or '')}</div>
      </div>
    </details>
  </div>

  <h2 style="margin-bottom:4px">All KPIs for this period</h2>
  <div class="sub" style="margin-bottom:11px">Click a card to make that KPI the subject of the analysis.</div>
  <div class="grid g4" style="margin-bottom:16px">{kpis}</div>

  <div class="grid g2" style="margin-bottom:16px">
    <div class="card pad">
      <h3>{esc(obs['kpi_label'])} by quarter</h3>
      <div class="sub" style="margin-bottom:8px">The shaded band is the range this KPI normally moves within for this comparison.</div>
      {trend_svg(obs)}
    </div>
    <div class="card pad">
      <h3>What drove the change</h3>
      <div class="sub" style="margin-bottom:10px">Ranked by how much each part of the business moved the KPI relative to its own size.</div>
      {drivers}
    </div>
  </div>

  <div class="card pad" style="margin-bottom:20px">
    <h3>Contribution by {esc(first_dim or '')}</h3>
    <div class="sub" style="margin-bottom:6px">Contributions sum to 100% of the change within the dimension.</div>
    <div class="legend" style="margin-bottom:6px">
      <span><i class="swatch" style="background:var(--diverge-neg)"></i>Pulled the KPI down</span>
      <span><i class="swatch" style="background:var(--diverge-pos)"></i>Pushed it up</span></div>
    {driver_chart}
  </div>

  <div class="eyebrow">Four-stage investigation</div>
  <h2>Observe &rarr; Investigate &rarr; Contest &rarr; Act</h2>
  <div class="sub">Each stage answers a different question. They are kept separate on purpose: proposing an explanation and trying to break it are not the same job.</div>

  <div class="stages">
    <div class="stage"><div class="n">Stage 1</div><div class="t">Observe</div><div class="q">What actually changed?</div></div>
    <div class="stage"><div class="n">Stage 2</div><div class="t">Investigate</div><div class="q">What could explain it?</div></div>
    <div class="stage on"><div class="n">Stage 3</div><div class="t">Contest</div><div class="q">What would disprove it?</div></div>
    <div class="stage"><div class="n">Stage 4</div><div class="t">Act</div><div class="q">What should we do?</div></div>
  </div>

  <div class="card pad" style="margin-bottom:14px">
    <div class="eyebrow">Stage 2 — Investigate</div>
    <h3 style="margin-top:3px">{len(inv['hypotheses'])} competing explanations, from {inv['considered_count']} considered</h3>
    <div class="sub">{esc(inv['method_note'])}</div>
    <div class="row" style="flex-wrap:wrap;margin-top:11px">
      {tone_chip(str(inv['documents_indexed']) + ' document chunks indexed','neutral')}{focus_chips}
    </div>
  </div>
  {hyp_summary}

  <div class="card pad" style="margin:18px 0 14px">
    <div class="eyebrow">Stage 3 — Contest</div>
    <h3 style="margin-top:3px">Ranked by evidence, after being challenged</h3>
    <div class="sub" style="margin-bottom:8px">Each explanation was tested for timing, consistency across the business, counterexamples and contradictory documents.</div>
    <table><tbody>{ranking}</tbody></table>
    <div class="muted" style="font-size:11.5px;margin-top:10px">{esc(con['confidence_disclaimer'])}</div>
  </div>
  {''.join(cards)}

  <div class="card pad" style="margin:18px 0 14px">
    <div class="eyebrow">Stage 4 — Act</div>
    <h3 style="margin-top:3px">{esc(n['headline'])}</h3>
    <div class="sub">{esc(n['what_changed'])}</div>
    <div class="sub">{esc(n['how_significant'])}</div>
    <div class="sub">{esc(n['what_drove_it'])}</div>
    <div class="sub">{esc(n['leading_explanation'])}</div>
  </div>
  <div class="callout warn" style="margin-bottom:14px"><b>What we cannot yet say</b>
    <ul class="clean">{''.join(f'<li>{esc(u)}</li>' for u in n['what_we_are_not_sure_about'])}</ul></div>

  <h2 style="margin-bottom:4px">Recommended next steps</h2>
  <div class="sub" style="margin-bottom:11px">Each is tied to the evidence that justifies it, and states what would change it.</div>
  {recs}

  <div class="card pad">
    <div class="eyebrow">Limits of this analysis</div>
    <ul class="clean">{''.join(f'<li>{esc(l)}</li>' for l in act['limits'])}</ul>
  </div>
</main>
</div>
</body></html>"""


def main() -> None:
    if not SRC.exists():
        raise SystemExit("Run scripts/run_pipeline_demo.py first to produce demo_output/investigation.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(json.loads(SRC.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
