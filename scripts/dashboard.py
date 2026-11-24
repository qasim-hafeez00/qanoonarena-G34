"""Step 4 dashboard: the four versions side by side on score, tokens, speed and wrong citations.

    python scripts/dashboard.py --group G34 --split dev
      -> report/dashboard_<split>.html  (open in a browser and screenshot it for the submission)
      -> report/dashboard_<split>.png   (same panels, for the LaTeX report)

Every number is recomputed from the unchanged results/*.csv files with the harness's own
aggregate(), as mean +/- standard deviation over the seeds. Wrong citations = share of
scored turns with at least one ungrounded citation AFTER the citation checker; the raw
model behaviour before the checker is read from the cite_check spans in traces/.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.common import load_config  # noqa: E402
from harness.metrics import aggregate  # noqa: E402

ORDER = ["M0+A0", "M1+A0", "M0+A1", "M1+A1"]
NAMES = {"M0+A0": "V1 baseline", "M1+A0": "V2 memory", "M0+A1": "V3 training", "M1+A1": "V4 both"}
COLORS = {"M0+A0": "#2a78d6", "M1+A0": "#eb6834", "M0+A1": "#1baf7a", "M1+A1": "#eda100"}  # reference palette slots 1-4
PANELS = [  # key, title, unit/format, higher_is_better
    ("score", "Composite score (0-100)", "{:.1f}", True),
    ("accuracy", "Answer accuracy", "{:.2f}", True),
    ("citation_f1", "Citation F1", "{:.2f}", True),
    ("memory_recall", "Memory recall (memory turns)", "{:.2f}", True),
    ("tokens_per_turn", "Tokens per turn (in + out)", "{:.0f}", False),
    ("latency_p50_ms", "Latency p50 (ms)", "{:.0f}", False),
    ("ungrounded_rate", "Turns with a wrong citation (after checker)", "{:.3f}", False),
    ("raw_ungrounded_rate", "Turns with a wrong citation (raw model)", "{:.3f}", False),
]


def load(group, split):
    cfg = load_config()
    res = defaultdict(dict)
    for fp in sorted((ROOT / "results").glob(f"{group}_*_{split}_s*.csv")):
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        if rows:
            res[rows[0]["config"]][int(rows[0]["seed"])] = rows
    base = None
    if "M0+A0" in res:
        base = st.mean(aggregate(r, cfg["scoring"]["weights"])["tokens_per_turn"] for r in res["M0+A0"].values())
    out = {}
    for c, seeds in res.items():
        per_seed = {s: aggregate(r, cfg["scoring"]["weights"], base) for s, r in seeds.items()}
        for s in per_seed:
            per_seed[s]["raw_ungrounded_rate"] = raw_ungrounded(group, c, split, s)
        out[c] = per_seed
    return out


def raw_ungrounded(group, config, split, seed):
    """Share of turns whose cite_check span saw >= 1 ungrounded citation (before dropping)."""
    files = sorted((ROOT / "traces").glob(f"{group}_{config}_{split}_s{seed}_*.jsonl"))
    if not files:
        return float("nan")
    n = bad = 0
    for line in open(files[-1], encoding="utf-8"):
        sp = json.loads(line)
        if sp["name"] == "cite_check":
            n += 1
            bad += 1 if sp["attributes"].get("n_ungrounded", 0) else 0
    return bad / n if n else float("nan")


def stats(per_seed, key):
    vals = [v[key] for v in per_seed.values() if v.get(key) == v.get(key)]  # drop NaN
    if not vals:
        return float("nan"), 0.0, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def svg_panel(data, key, title, fmt, better):
    W, H, L, B, T = 330, 190, 8, 34, 30
    cfgs = [c for c in ORDER if c in data]
    vals = {c: stats(data[c], key) for c in cfgs}
    top = max([m + s for m, s, n in vals.values() if m == m] + [1e-9]) * 1.15
    bw, gap = 54, 22
    x0 = L + (W - L - (len(cfgs) * bw + (len(cfgs) - 1) * gap)) / 2
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{L}" y="18" class="t">{html.escape(title)} '
             f'<tspan class="m">{"higher" if better else "lower"} is better</tspan></text>',
             f'<line x1="{L}" x2="{W - 4}" y1="{H - B}" y2="{H - B}" class="axis"/>']
    ph = H - B - T
    for i, c in enumerate(cfgs):
        m, s, n = vals[c]
        x = x0 + i * (bw + gap)
        if m != m:
            parts.append(f'<text x="{x + bw / 2}" y="{H - B - 6}" class="v" text-anchor="middle">n/a</text>')
        else:
            h = ph * m / top
            y = H - B - h
            r = min(4, h / 2)
            tip = f"{NAMES[c]} ({c}): {fmt.format(m)} ± {fmt.format(s)} over {n} seed(s)"
            parts.append(f'<g class="bar"><title>{html.escape(tip)}</title>'
                         f'<rect x="{x - 6}" y="{T}" width="{bw + 12}" height="{ph}" fill="transparent"/>'
                         f'<path d="M{x},{H - B} V{y + r} Q{x},{y} {x + r},{y} H{x + bw - r} Q{x + bw},{y} {x + bw},{y + r} V{H - B} Z" fill="{COLORS[c]}"/>')
            if s > 0:
                e1, e2 = H - B - ph * (m - s) / top, H - B - ph * (m + s) / top
                parts.append(f'<line x1="{x + bw / 2}" x2="{x + bw / 2}" y1="{e1}" y2="{e2}" class="err"/>'
                             f'<line x1="{x + bw / 2 - 6}" x2="{x + bw / 2 + 6}" y1="{e2}" y2="{e2}" class="err"/>'
                             f'<line x1="{x + bw / 2 - 6}" x2="{x + bw / 2 + 6}" y1="{e1}" y2="{e1}" class="err"/>')
            parts.append(f'<text x="{x + bw / 2}" y="{min(e2 if s > 0 else y, y) - 5}" class="v" text-anchor="middle">{fmt.format(m)}</text></g>')
        parts.append(f'<text x="{x + bw / 2}" y="{H - B + 15}" class="lab" text-anchor="middle">{c}</text>')
        parts.append(f'<text x="{x + bw / 2}" y="{H - B + 28}" class="lab2" text-anchor="middle">{NAMES[c].split(" ")[0]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_html(data, group, split, out):
    cfgs = [c for c in ORDER if c in data]
    legend = "".join(f'<span class="key"><i style="background:{COLORS[c]}"></i>{NAMES[c]} ({c})</span>' for c in cfgs)
    panels = "".join(f'<div class="card">{svg_panel(data, k, t, f, b)}</div>' for k, t, f, b in PANELS)
    head = "".join(f"<th>{html.escape(t)}</th>" for _, t, _, _ in PANELS)
    rows = ""
    for c in cfgs:
        cells = ""
        for k, _, f, _ in PANELS:
            m, s, n = stats(data[c], k)
            cells += f"<td>{'n/a' if m != m else f.format(m) + ' ± ' + f.format(s)}</td>"
        rows += f"<tr><th>{NAMES[c]}<br><small>{c} · seeds {sorted(data[c])}</small></th>{cells}</tr>"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>QanoonArena {group} dashboard ({split})</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835}}}}
body{{margin:24px;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:var(--ink2);margin:0 0 12px;max-width:900px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 16px}} .key i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.card{{border:1px solid var(--rule);border-radius:10px;padding:8px}}
svg{{width:100%;height:auto}} .t{{font-size:12.5px;font-weight:600;fill:var(--ink)}} .m{{font-weight:400;fill:var(--ink2);font-size:11px}}
.v{{font-size:11.5px;fill:var(--ink)}} .lab{{font-size:11px;fill:var(--ink)}} .lab2{{font-size:10.5px;fill:var(--ink2)}}
.axis{{stroke:var(--rule);stroke-width:1}} .err{{stroke:var(--ink2);stroke-width:1.5}}
.bar:hover path{{opacity:.85}}
table{{border-collapse:collapse;margin-top:20px;font-size:12.5px}} td,th{{border-bottom:1px solid var(--rule);padding:6px 10px;text-align:right}} th:first-child{{text-align:left}}
</style></head><body>
<h1>QanoonArena · Group {group} · M1 summary memory × A1 QLoRA SFT · split: {split}</h1>
<p>Each bar is the mean over the fixed seeds; whiskers show ± one standard deviation across seeds. Hover a bar for exact values. Numbers are recomputed from the unchanged results/*.csv files with harness.metrics.aggregate.</p>
<div class="legend">{legend}</div>
<div class="grid">{panels}</div>
<table><thead><tr><th>Version</th>{head}</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    out.write_text(doc, encoding="utf-8")


def write_png(data, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped the PNG")
        return
    cfgs = [c for c in ORDER if c in data]
    keys = PANELS[:2] + PANELS[2:3] + PANELS[4:7]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (k, t, f, b) in zip(axes.flat, keys):
        ms = [stats(data[c], k) for c in cfgs]
        ax.bar(range(len(cfgs)), [m for m, _, _ in ms], yerr=[s for _, s, _ in ms], capsize=4,
               color=[COLORS[c] for c in cfgs], width=0.6, error_kw={"elinewidth": 1.2, "ecolor": "#52514e"})
        for i, (m, s, _) in enumerate(ms):
            if m == m:
                ax.text(i, m + s, f.format(m), ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(cfgs)), cfgs, fontsize=8.5)
        ax.set_title(t + (" ↑" if b else " ↓"), fontsize=10)
        ax.set_ylim(bottom=0, top=max([m + s for m, s, _ in ms if m == m] + [0]) * 1.15 or 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Four versions, mean ± sd over seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="G34")
    p.add_argument("--split", default="dev")
    a = p.parse_args()
    data = load(a.group, a.split)
    if not data:
        sys.exit("no results found")
    (ROOT / "report").mkdir(exist_ok=True)
    write_html(data, a.group, a.split, ROOT / "report" / f"dashboard_{a.split}.html")
    write_png(data, ROOT / "report" / f"dashboard_{a.split}.png")
    print(f"wrote report/dashboard_{a.split}.html and .png")


if __name__ == "__main__":
    main()
