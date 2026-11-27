"""Run this Colab account's share of the G34 experiments, several processes on one T4. Resumable.

    python scripts/colab_jobs.py              # role picked automatically
    python scripts/colab_jobs.py --role main  # or force it

Roles (auto: "main" if this Drive holds the A1 adapter or a training checkpoint, else "helper"):
  main    dev M0+A1 and M1+A1 (needs adapters/a1), then the MEMORY.md demo runs
  helper  dev M0+A0 and M1+A0, and the memory ablation (G34abl40, QA_M1_MAX_WORDS=40)
  test    secret test, all 4 versions (needs --questions; A1 versions only if the adapter is here)

Each job is scripts/g34_run.py (the fixed harness evaluator, one (version, seed) at a time,
finished runs skipped), so a lost runtime only repeats the run that was in progress.
Inference needs ~3.5 GB, so 2-3 processes share a 15 GB T4 (see DEVIATIONS.md: latency only).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AD = "adapters/a1"


def has_adapter() -> bool:
    return (ROOT / AD / "adapter_config.json").exists()


def jobs_for(role: str, questions: str | None) -> list[list[list[str]]]:
    """Each job is a list of g34_run argument lists, run one after another."""
    run = ["--split", "dev"]
    if role == "main":
        return [[run + ["--configs", "M0+A1", "--adapter-path", AD]],
                [run + ["--configs", "M1+A1", "--adapter-path", AD],
                 ["--split", "demo", "--group", "G34demo", "--questions", "report/memory_demo.jsonl",
                  "--configs", "M1+A0,M1+A1", "--seeds", "13", "--adapter-path", AD]]]
    if role == "helper":
        return [[run + ["--configs", "M0+A0"]],
                [run + ["--configs", "M1+A0"]],
                [run + ["--group", "G34abl40", "--configs", "M1+A0", "--env", "QA_M1_MAX_WORDS=40"]]]
    if role == "test":
        if not questions:
            sys.exit("--role test needs --questions")
        t = ["--split", "test", "--questions", questions]
        a1 = ["--adapter-path", AD] if has_adapter() else []
        jobs = [[t + ["--configs", "M0+A0"]], [t + ["--configs", "M1+A0"]]]
        if a1:
            jobs += [[t + ["--configs", "M0+A1"] + a1], [t + ["--configs", "M1+A1"] + a1]]
        return jobs
    sys.exit(f"unknown role {role}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", default="auto")
    p.add_argument("--questions", default=None)
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--logs", default="outputs/logs")  # outputs/ is on Drive
    a = p.parse_args(argv)
    role = a.role
    if role == "auto":
        role = "main" if has_adapter() or any((ROOT / "outputs" / "sft").glob("checkpoint-*")) else "helper"
    if role == "main" and not has_adapter():
        sys.exit("role main: the adapter is not trained yet (run agent.training.train_sft first)")
    print("role:", role, flush=True)

    logs = ROOT / a.logs
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, steps in enumerate(jobs_for(role, a.questions)):
        cmds = " && ".join(" ".join([sys.executable, "scripts/g34_run.py", *map(_q, s)]) for s in steps)
        print(f"[job {i}] {cmds}", flush=True)
        if a.dry_run:
            continue
        log = open(logs / f"{role}_job{i}.txt", "a")
        procs.append(subprocess.Popen(cmds, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, PYTHONUNBUFFERED="1")))
        time.sleep(60)  # stagger model loading

    if a.dry_run:
        return
    t0 = time.time()
    while any(pr.poll() is None for pr in procs):
        time.sleep(300)
        tails = []
        for i in range(len(procs)):
            txt = (logs / f"{role}_job{i}.txt").read_text(errors="ignore")[-300:].replace("\r", "\n")
            tails.append(f"job{i}: " + (txt.strip().splitlines() or [""])[-1][:100])
        print(f"[{(time.time() - t0) / 60:.0f} min] " + " | ".join(tails), flush=True)
    codes = [pr.returncode for pr in procs]
    print("exit codes:", codes, flush=True)
    if any(codes):
        sys.exit(1)


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
