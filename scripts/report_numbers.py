"""Build every number the report and model card need, from the unchanged results/traces files.

    python scripts/report_numbers.py --group G34 --split dev \
        --ablation-group G34abl40 --errors ERRORS.csv --train-meta adapters/a1/training_meta.json

Writes report/generated/*.tex (rows that \\input into report/section.tex) and prints a Markdown
summary for MODEL_CARD.md / README. It only reads files; it never edits results.

Numbers:
  main      harness.summarize --latex (the official table; mean of seeds, 95% bootstrap CI)
  cost      per version, mean over seeds: answer tokens/turn (as scored), M1 summariser
            tokens/turn (from memory_write spans, which the evaluator does not count),
            latency p50/p95, wall-clock per run, training GPU-hours
  ablation  score / memory recall / tokens per turn for the 40-word summary, the default and M0
  errors    count of labelled failures per version and code (ERRORS.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["M0+A0", "M1+A0", "M0+A1", "M1+A1"]
CODES = [f"E{i}" for i in range(1, 9)]


def manifests(group: str, split: str) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for fp in sorted((ROOT / "results").glob(f"{group}_*_{split}_s*.json")):
        m = json.loads(fp.read_text())
        if m.get("summary"):
            out[m["config"]].append(m)
    return out


def latest_trace(stem: str) -> Path | None:
    fps = sorted((ROOT / "traces").glob(f"{stem}_*.jsonl"))
    return fps[-1] if fps else None


def summariser_tokens(group: str, cfg: str, split: str, seed: int) -> tuple[float, int] | None:
    """Mean (tokens_in + tokens_out) of the M1 summary rewrite per turn, and the number of turns."""
    fp = latest_trace(f"{group}_{cfg}_{split}_s{seed}")
    if fp is None:
        return None
    total, turns = 0, 0
    with open(fp, encoding="utf-8") as f:
        for line in f:
            sp = json.loads(line)
            if sp.get("name") == "turn":
                turns += 1
            elif sp.get("name") == "memory_write":
                for it in sp.get("attributes", {}).get("items", []):
                    if it.get("op") == "summary":
                        total += int(it.get("tokens_in") or 0) + int(it.get("tokens_out") or 0)
    return (total / turns if turns else 0.0), turns


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def fmt(x, nd=0):
    return "--" if x != x else f"{x:,.{nd}f}".replace(",", "{,}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="G34")
    p.add_argument("--split", default="dev")
    p.add_argument("--ablation-group", default="G34abl40")
    p.add_argument("--errors", default="ERRORS.csv")
    p.add_argument("--train-meta", default="adapters/a1/training_meta.json")
    p.add_argument("--out", default="report/generated")
    a = p.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    md = []

    # ---- main table (official) -------------------------------------------------------
    r = subprocess.run([sys.executable, "-m", "harness.summarize", "--group", a.group, "--split", a.split,
                        "--latex"], cwd=ROOT, capture_output=True, text=True)
    (out / f"main_{a.split}.tex").write_text(r.stdout)
    r2 = subprocess.run([sys.executable, "-m", "harness.summarize", "--group", a.group, "--split", a.split],
                        cwd=ROOT, capture_output=True, text=True)
    md += ["## Main results (harness.summarize)", "```", r2.stdout.strip(), "```"]

    # ---- training cost ----------------------------------------------------------------
    gpu_h, gpu = float("nan"), "?"
    tm = ROOT / a.train_meta
    if tm.exists():
        meta = json.loads(tm.read_text())
        gpu = meta.get("gpu", "?")
        hist = meta.get("log_history", [])
        rt = [x.get("train_runtime") for x in hist if "train_runtime" in x]
        minutes = (rt[-1] / 60) if rt else meta.get("train_minutes_this_session", float("nan"))
        gpu_h = minutes / 60
        losses = [x for x in hist if "loss" in x]
        md += ["## Training", f"- GPU: {gpu}", f"- train_runtime: {minutes:.1f} min = {gpu_h:.2f} GPU-hours",
               f"- loss first/last logged: {losses[0]['loss']:.4f} -> {losses[-1]['loss']:.4f}" if losses else "- no loss logged",
               f"- steps: {losses[-1]['step'] if losses else '?'}"]

    # ---- cost table ---------------------------------------------------------------------
    ms = manifests(a.group, a.split)
    rows, md_cost = [], ["## Cost (mean over seeds)", "| Version | answer tok/turn | summariser tok/turn | p50 ms | p95 ms | wall-clock min/run | train GPU-h |", "|---|---|---|---|---|---|---|"]
    for cfg in ORDER:
        runs = ms.get(cfg, [])
        if not runs:
            continue
        tok = mean([m["summary"]["tokens_per_turn"] for m in runs])
        p50 = mean([m["summary"]["latency_p50_ms"] for m in runs])
        p95 = mean([m["summary"]["latency_p95_ms"] for m in runs])
        wall = mean([m.get("wall_clock_s", 0) / 60 for m in runs])
        summ = mean([(summariser_tokens(a.group, cfg, a.split, m["seed"]) or (None, 0))[0] for m in runs]) if cfg.startswith("M1") else 0.0
        th = gpu_h if cfg.endswith("A1") else 0.0
        rows.append(f"{cfg} & {fmt(tok)} & {fmt(summ)} & {fmt(p50)} & {fmt(p95)} & {fmt(th, 2)} \\\\")
        md_cost.append(f"| {cfg} | {tok:.0f} | {summ:.0f} | {p50:.0f} | {p95:.0f} | {wall:.1f} | {th:.2f} |")
    (out / f"cost_{a.split}.tex").write_text("\n".join(rows) + "\n")
    md += md_cost

    # ---- ablation -----------------------------------------------------------------------
    ab = manifests(a.ablation_group, a.split)
    abl_rows, md_ab = [], ["## Memory ablation (summary target words)", "| Setting | n seeds | score | memory recall | tokens/turn |", "|---|---|---|---|---|"]
    for label, runs in [("Summary target 40 words", ab.get("M1+A0", [])),
                        ("Summary target 120 words (default)", ms.get("M1+A0", [])),
                        ("M0 (no memory)", ms.get("M0+A0", []))]:
        if not runs:
            continue
        sc = [m["summary"]["score"] for m in runs]
        mr = mean([m["summary"]["memory_recall"] for m in runs])
        tk = mean([m["summary"]["tokens_per_turn"] for m in runs])
        sd = st.pstdev(sc) if len(sc) > 1 else 0.0
        abl_rows.append(f"{label} & {mean(sc):.2f} $\\pm$ {sd:.2f} & {mr:.3f} & {fmt(tk)} \\\\")
        md_ab.append(f"| {label} | {len(sc)} | {mean(sc):.2f} ± {sd:.2f} | {mr:.3f} | {tk:.0f} |")
    (out / f"ablation_{a.split}.tex").write_text("\n".join(abl_rows) + "\n")
    md += md_ab

    # ---- errors -------------------------------------------------------------------------
    ep = ROOT / a.errors
    if ep.exists():
        cnt = defaultdict(Counter)
        with open(ep, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cnt[row["config"]][row["code"].strip()] += 1
        err_rows = [f"{cfg} & " + " & ".join(str(cnt[cfg][c]) for c in CODES) + " \\\\" for cfg in ORDER if cnt[cfg]]
        (out / "errors.tex").write_text("\n".join(err_rows) + "\n")
        md += ["## Labelled errors", "| Version | " + " | ".join(CODES) + " | total |", "|---" * 10 + "|"]
        md += [f"| {cfg} | " + " | ".join(str(cnt[cfg][c]) for c in CODES) + f" | {sum(cnt[cfg].values())} |"
               for cfg in ORDER if cnt[cfg]]

    print("\n".join(md))


if __name__ == "__main__":
    main()
