"""Resumable experiment runner for Group G34 (M1 summary memory x A1 SFT).

It only calls the fixed harness evaluator (python -m harness.evaluate) once per
(version, seed), skipping runs that already finished, so a Colab disconnect costs at most
the run in progress. It never edits a results file.

    # all 4 versions x 3 seeds on the practice set
    python scripts/g34_run.py --split dev --adapter-path adapters/a1

    # split the work between group members (each on their own Colab):
    python scripts/g34_run.py --split dev --adapter-path adapters/a1 --configs M0+A0,M1+A0
    python scripts/g34_run.py --split dev --adapter-path adapters/a1 --configs M0+A1,M1+A1

    # secret test (Arena window)
    python scripts/g34_run.py --split test --questions secret_test_questions.jsonl --adapter-path adapters/a1

    # memory ablation (Step 5.2): shorter summaries, kept under a separate group id
    python scripts/g34_run.py --split dev --group G34abl40 --configs M1+A0 --env QA_M1_MAX_WORDS=40

Run M0+A0 first: the evaluator reads its token count to compute the efficiency score.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["M0+A0", "M1+A0", "M0+A1", "M1+A1"]


def done(group: str, cfg: str, split: str, seed: int, backend: str) -> bool:
    fp = ROOT / "results" / f"{group}_{cfg}_{split}_s{seed}.json"
    if not fp.exists():
        return False
    try:
        man = json.loads(fp.read_text())
    except json.JSONDecodeError:
        return False
    return man.get("backend") == backend and bool(man.get("summary"))


def clean_partial_traces(group: str, cfg: str, split: str, seed: int) -> None:
    for fp in (ROOT / "traces").glob(f"{group}_{cfg}_{split}_s{seed}_*.jsonl"):
        print(f"  removing partial trace from an unfinished run: {fp.name}")
        fp.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="G34")
    p.add_argument("--split", default="dev")
    p.add_argument("--questions", default=None)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--configs", default=",".join(ORDER))
    p.add_argument("--seeds", default="13,42,2026")
    p.add_argument("--backend", default="hf")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for memory ablations")
    a = p.parse_args(argv)

    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    configs = [c.strip().upper() for c in a.configs.split(",") if c.strip()]
    configs.sort(key=lambda c: ORDER.index(c) if c in ORDER else 99)
    seeds = [int(s) for s in a.seeds.split(",")]

    for cfg in configs:
        mem, ad = cfg.split("+")
        for seed in seeds:
            if done(a.group, cfg, a.split, seed, a.backend) and not a.limit:
                print(f"[skip] {a.group} {cfg} {a.split} seed {seed}: already finished")
                continue
            clean_partial_traces(a.group, cfg, a.split, seed)
            cmd = [sys.executable, "-m", "harness.evaluate", "--group", a.group, "--memory", mem,
                   "--adapter", ad, "--split", a.split, "--seed", str(seed), "--backend", a.backend]
            if ad != "A0":
                if not a.adapter_path:
                    sys.exit(f"{cfg} needs --adapter-path")
                cmd += ["--adapter-path", a.adapter_path]
            if a.questions:
                cmd += ["--questions", a.questions]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            print("[run]", " ".join(cmd), " ".join(a.env))
            r = subprocess.run(cmd, cwd=ROOT, env=env)
            if r.returncode != 0:
                sys.exit(f"run failed: {cfg} seed {seed} (exit {r.returncode}); re-run this script to resume")


if __name__ == "__main__":
    main()
