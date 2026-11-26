"""Run G34 experiments in parallel: one version per GPU (Kaggle 2x T4), or several on one GPU
("0=M0+A0;0=M1+A0": inference needs ~3.5 GB, so two processes share a 15 GB T4).

Kaggle notebook (Accelerator: GPU T4 x2, Internet: on), one cell, then "Save Version > Save & Run All":

    !git clone -q https://github.com/qasim-hafeez00/qanoonarena-G34.git /kaggle/working/qa
    %cd /kaggle/working/qa
    !python scripts/kaggle_job.py --jobs "0=M0+A0;1=M1+A0" --split dev

    # A1 versions need the trained adapter (shared Drive folder link or Hugging Face id):
    !python scripts/kaggle_job.py --jobs "0=M0+A1;1=M1+A1" --split dev --adapter-gdrive <folder link>

Each job is scripts/g34_run.py (which only calls the fixed harness evaluator) with
CUDA_VISIBLE_DEVICES set to that GPU. At the end, results/ and traces/ are zipped to
/kaggle/working/G34_<tag>.zip, which Kaggle keeps as the notebook output.
Nothing in harness/ or config.yaml changes; the same pinned packages as the Colab notebook.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = ["transformers>=4.51,<5", "bitsandbytes>=0.43", "peft>=0.13", "accelerate>=0.30",
        "langgraph>=0.4,<2", "pyyaml>=6", "hf_xet"]


def sh(cmd: list[str], **kw) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch_adapter(a) -> str | None:
    dest = ROOT / "adapters" / "a1"
    if (dest / "adapter_config.json").exists():
        return str(dest)
    if a.adapter_gdrive:
        sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        sh([sys.executable, "-m", "gdown", "--folder", a.adapter_gdrive, "-O", str(dest)])
        # gdown may nest the folder one level down
        if not (dest / "adapter_config.json").exists():
            sub = [p for p in dest.iterdir() if (p / "adapter_config.json").exists()]
            if sub:
                return str(sub[0])
        return str(dest)
    if a.adapter_hf:
        return a.adapter_hf
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", required=True, help='"GPU=CFG[,CFG];GPU=CFG", e.g. "0=M0+A0;1=M1+A0"')
    p.add_argument("--split", default="dev")
    p.add_argument("--group", default="G34")
    p.add_argument("--seeds", default="13,42,2026")
    p.add_argument("--questions", default=None)
    p.add_argument("--adapter-gdrive", default=None)
    p.add_argument("--adapter-hf", default=None)
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else str(ROOT / "outputs"))
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--backend", default="hf", help="mock only for testing this script")
    p.add_argument("--limit", type=int, default=0, help="testing only")
    a = p.parse_args(argv)

    if not a.no_install:
        sh([sys.executable, "-m", "pip", "install", "-q", *PKGS])
    adapter = None
    if "A1" in a.jobs.upper():
        adapter = fetch_adapter(a)
        if not adapter:
            sys.exit("A1 jobs need --adapter-gdrive or --adapter-hf")
        print("adapter:", adapter, flush=True)

    procs = []
    logs = Path(a.out_dir)
    logs.mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(a.jobs.split(";")):
        gpu, _, cfgs = job.partition("=")
        cmd = [sys.executable, "scripts/g34_run.py", "--group", a.group, "--split", a.split,
               "--configs", cfgs, "--seeds", a.seeds, "--backend", a.backend]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if adapter:
            cmd += ["--adapter-path", adapter]
        if a.questions:
            cmd += ["--questions", a.questions]
        for e in a.env:
            cmd += ["--env", e]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.strip(), PYTHONUNBUFFERED="1")
        log = open(logs / f"log_job{i}_gpu{gpu.strip()}.txt", "w")
        print(f"[gpu {gpu}]", " ".join(cmd), flush=True)
        procs.append((gpu, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(90 if a.backend == "hf" else 1)  # stagger model loading so the two processes do not download at once

    t0 = time.time()
    while any(pr.poll() is None for _, pr, _ in procs):
        time.sleep(300 if a.backend == "hf" else 5)
        done = sorted(x.name for x in (ROOT / "results").glob(f"{a.group}_*_{a.split}_s*.json"))
        print(f"[{(time.time() - t0) / 60:.0f} min] finished runs: {done}", flush=True)
    codes = {g: pr.returncode for g, pr, _ in procs}
    print("exit codes:", codes, flush=True)

    tag = a.tag or f"{a.group}_{a.split}_" + a.jobs.replace("=", "-").replace(";", "_").replace("+", "").replace(",", "-")
    zp = Path(a.out_dir) / f"{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("results", "traces"):
            for fp in (ROOT / sub).rglob("*"):
                if fp.is_file():
                    z.write(fp, fp.relative_to(ROOT))
        for fp in logs.glob("log_job*_gpu*.txt"):
            z.write(fp, f"logs/{fp.name}")
    print("wrote", zp, flush=True)
    if any(codes.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
