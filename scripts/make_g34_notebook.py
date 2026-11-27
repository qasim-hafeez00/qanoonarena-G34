"""Writes notebooks/G34_colab_pipeline.ipynb (run once: python scripts/make_g34_notebook.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# G34 QanoonArena pipeline: M1 summary memory × A1 QLoRA SFT (Colab T4)

**Runtime > Run all** (T4 GPU) and approve Google Drive. That is all: every cell checks what is already
on Drive (`results/`, `traces/`, `adapters/`, `outputs/`) and skips or resumes it, so after a lost runtime
you just do Run all again. The account whose Drive holds the training checkpoint/adapter is the **main**
account (training, then the A1 versions); any other account is a **helper** (A0 versions + memory ablation).

| Part | What | Approx. T4 time |
|---|---|---|
| 1 | setup + Drive | 5 min |
| 2 | SFT data (harness builder → fit to 2048 tokens → leakage check) | 5 min |
| 3 | QLoRA SFT (A1) + loss curve + upload | 2–4 h |
| 4 | experiments on dev: 4 versions × 3 seeds (resumable; split across members) | 8–12 h total |
| 5 | memory ablation | 2–3 h |
| 6 | secret test (Arena window) | like part 4 |
| 7 | smoke regression gate, trace bundle, dashboard | 30 min |
""")
md("## 1. Setup")
code("""
# ---- change these -----------------------------------------------------------
REPO_URL = "https://github.com/qasim-hafeez00/qanoonarena-G34.git"   # group repo (if private: use a token URL)
GROUP    = "G34"
HF_REPO  = ""          # e.g. "your-hf-user/qanoonarena-G34-A1" to upload the adapter (optional)
DRIVE_DIR = "/content/drive/MyDrive/qanoonarena_G34"
# -----------------------------------------------------------------------------
import os, subprocess
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)
if not os.path.exists("/content/qanoonarena"):
    subprocess.run(["git", "clone", REPO_URL, "/content/qanoonarena"], check=True)
os.chdir("/content/qanoonarena")
subprocess.run(["git", "pull"], check=False)
# keep the outputs on Drive (survive disconnects)
for d in ["results", "traces", "adapters", "submission", "outputs"]:
    target = f"{DRIVE_DIR}/{d}"
    os.makedirs(target, exist_ok=True)
    if os.path.islink(d):
        continue
    if os.path.isdir(d):
        subprocess.run(f"cp -rn {d}/. {target}/ && rm -rf {d}", shell=True, check=True)
    os.symlink(target, d)
print(os.getcwd(), os.listdir("."))
""")
code("""
# transformers < 5: the fixed harness/backends.py reads `ids.shape` from apply_chat_template,
# which transformers 5 turned into a dict (see DEVIATIONS.md). Run on a fresh runtime (Run all).
!pip install -q -r requirements-train.txt "transformers>=4.51,<5" hf_xet matplotlib
import torch, transformers; print(torch.cuda.get_device_name(0), "transformers", transformers.__version__)
""")
md("""### 1b. Live traces in Arize Phoenix (Step 4)
The harness always writes `traces/*.jsonl` (the file we submit). This cell also mirrors every span
to a local Arize Phoenix server through OpenTelemetry (`harness/tracing.py` reads
`OTEL_EXPORTER_OTLP_ENDPOINT`). Run it after every runtime restart, before any experiment; open the
printed URL to browse traces and take the screenshots for the report.
""")
code("""
!pip install -q arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
import os, phoenix as px
session = px.launch_app()
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:6006"   # child processes inherit it
print("Phoenix UI:", session.url)
""")
md("""## 2. Training (Step 3, main account only)
Data: the harness builder makes prompts exactly as the bot sees them; `agent.training.prepare_sft`
shortens the DOCUMENTS block of the 215 prompts longer than the fixed `max_seq_len` (2048) so the
answer is never truncated; the leakage check must say OK. Training: `agent.training.train_sft`
(QLoRA, fixed budget from config.yaml) in a subprocess, so the GPU is free again afterwards. It resumes
from the newest checkpoint in `outputs/sft/` and is skipped once `adapters/a1` exists.""")
code("""
import glob, os, subprocess, sys
HAS_ADAPTER = os.path.exists("adapters/a1/adapter_config.json")
MAIN = HAS_ADAPTER or bool(glob.glob("outputs/sft/checkpoint-*"))
print("adapter trained:", HAS_ADAPTER, "| this is the", "MAIN" if MAIN else "HELPER", "account")
if MAIN and not HAS_ADAPTER:
    for cmd in ["python -m harness.build_sft_data --out data_sft.jsonl",
                "python -m agent.training.prepare_sft data_sft.jsonl data_sft_fit.jsonl",
                "python -m harness.leakage_check data_sft_fit.jsonl",
                "python -m agent.training.train_sft --data data_sft_fit.jsonl --out adapters/a1 --work outputs/sft"]:
        print("$", cmd, flush=True)
        subprocess.run(cmd, shell=True, check=True)
""")
code("""
import json, os, matplotlib.pyplot as plt
if os.path.exists("outputs/sft/loss_log.jsonl"):
    pts = {}
    for l in open("outputs/sft/loss_log.jsonl"):
        r = json.loads(l)
        if "loss" in r: pts[r["step"]] = r["loss"]
    s = sorted(pts)
    plt.figure(figsize=(6, 3.5)); plt.plot(s, [pts[k] for k in s], color="#2a78d6", lw=2)
    plt.xlabel("optimizer step"); plt.ylabel("training loss"); plt.title("G34 A1 QLoRA SFT loss"); plt.grid(alpha=.3)
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/sft_loss.png", dpi=160, bbox_inches="tight"); plt.show()
if os.path.exists("adapters/a1/training_meta.json"):
    m = json.load(open("adapters/a1/training_meta.json"))
    print(m["gpu"], m["train_minutes_total"], "min over", len(m["sessions"]), "session(s)")
""")
md("""## 3. Experiments on the practice set (Step 5)
`scripts/colab_jobs.py` starts this account's jobs side by side on the T4 (main: M0+A1, M1+A1, then the
MEMORY.md demo; helper: M0+A0, M1+A0 and the 40-word memory ablation). Finished (version, seed) runs are
skipped, so re-running resumes. It prints progress every 5 minutes.""")
code("""
!git pull -q; python scripts/colab_jobs.py
""")
code("""
!python -m harness.summarize --group G34 --split dev
!python -m harness.summarize --group G34abl40 --split dev
""")
md("## 4. Secret test (only inside the 48-hour Arena window; skipped until the file is in the repo folder)")
code("""
SECRET = "secret_test_questions.jsonl"
if os.path.exists(SECRET):
    !python scripts/colab_jobs.py --role test --questions {SECRET}
else:
    print("no", SECRET, "here: skipped")
""")
md("## 5. Regression gate (real model), trace bundle, dashboard, error candidates")
code("""
!python -m harness.smoke --group G34 --memory M1 --adapter A0
!python -m harness.trace_export --group G34
!python scripts/dashboard.py --group G34 --split dev
!python scripts/error_candidates.py --group G34 --split dev --n 80 > outputs/ERRORS_candidates.csv
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "G34_colab_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
