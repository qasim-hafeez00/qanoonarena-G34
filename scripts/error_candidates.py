"""Step 4 item 5 helper: list failed turns with a SUGGESTED error code, for a human to verify.

    python scripts/error_candidates.py --group G34 --split dev --n 60 > ERRORS_candidates.csv

The suggestion is only a starting point. Open each turn's trace (session_id + turn in
traces/<run>.jsonl) and confirm or change the code before copying the row to ERRORS.csv.
Rules, first match wins:
  E8  parse_ok == 0                                   (not valid JSON)
  E6  should abstain, did not                         (answered an unanswerable question)
  E7  abstained, should not                           (refused an answerable one)
  E1  a gold document was not among the retrieved ids
  E3  memory-dependent turn, gold doc retrieved, the answer misses the remembered fact
  E2  T4 (amendment) turn, wrong year/law
  E5  cited something that is not the gold source, or an ungrounded id
  E4  memory-dependent turn where the memory text contains a superseded/wrong value
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.common import read_jsonl  # noqa: E402

SPLITS = {"dev": "dev.jsonl", "smoke": "smoke.jsonl"}


def suggest(row, gold, pred, typ):
    g_cites = gold.get("citations") or []
    retrieved = pred.get("retrieved") or []
    cites = pred["answer"].get("citations") or []
    if row["parse_ok"] == "0":
        return "E8", "", "output was not valid JSON"
    if row["abstain_expected"] == "1" and row["abstained"] == "0":
        return "E6", ("E5" if cites else ""), f"answered with {cites or 'no citation'} instead of 'I am not sure'"
    if row["abstain_expected"] == "0" and row["abstained"] == "1":
        sec = "E1" if g_cites and not any(g in retrieved for g in g_cites) else ""
        return "E7", sec, f"refused; gold {g_cites} {'not ' if sec else ''}retrieved"
    if g_cites and not any(g in retrieved for g in g_cites):
        sec = "E3" if gold.get("memory_dependent") else ""
        return "E1", sec, f"gold {g_cites} not in retrieved {retrieved[:5]}"
    if gold.get("memory_dependent") and typ == "T3":
        return "E3", ("E5" if set(cites) - set(g_cites) else ""), \
            f"gold {g_cites} was retrieved but the answer missed keywords {gold.get('answer_keywords')}"
    if typ == "T4":
        return "E2", ("E5" if set(cites) - set(g_cites) else ""), \
            f"amendment answer missed {gold.get('answer_keywords')}"
    if set(cites) - set(g_cites) or int(row["ungrounded_cites"] or 0):
        return "E5", "", f"cited {cites}, gold {g_cites}"
    return "E5", "", f"answer did not contain {gold.get('answer_keywords')} (cited {cites})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="G34")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sessions = {s["session_id"]: s for s in read_jsonl(ROOT / "harness" / "data" / SPLITS[a.split])}
    by_cfg = defaultdict(list)
    for fp in sorted((ROOT / "results").glob(f"{a.group}_*_{a.split}_s*.csv")):
        preds = {(p["session_id"], p["turn"]): p
                 for p in read_jsonl(str(fp).replace(".csv", ".pred.jsonl"))}
        for row in csv.DictReader(open(fp, encoding="utf-8")):
            bad = row["correct"] == "0" or (row["cite_precision"] not in ("", "1.0") and row["abstain_expected"] == "0")
            if not bad:
                continue
            s = sessions[row["session_id"]]
            t = int(row["turn"])
            gold = s["turns"][t - 1]["gold"]
            code, sec, note = suggest(row, gold, preds[(row["session_id"], t)], s["type"])
            by_cfg[row["config"]].append([row["config"], row["seed"], row["session_id"], t, code, sec,
                                          "AUTO-CHECK: " + note])
    rng = random.Random(a.seed)
    out, cfgs = [], sorted(by_cfg)
    for rows in by_cfg.values():
        rng.shuffle(rows)
    while len(out) < a.n and any(by_cfg.values()):  # round-robin over the versions
        for c in cfgs:
            if by_cfg[c] and len(out) < a.n:
                out.append(by_cfg[c].pop())
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["config", "seed", "session_id", "turn", "code", "secondary_code", "note"])
    w.writerows(out)


if __name__ == "__main__":
    main()
