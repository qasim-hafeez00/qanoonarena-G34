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

Runtime > Change runtime type > **T4 GPU**. Run the cells top to bottom. Everything that matters
(`results/`, `traces/`, `adapters/`, `submission/`) lives on **Google Drive**, so a disconnect loses
at most the run in progress: just run the notebook again and it resumes.

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
REPO_URL = "https://github.com/<your-account>/<your-repo>.git"   # your group's repo (private is fine: use a token URL)
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
!pip install -q -r requirements-train.txt hf_xet matplotlib
import torch; print(torch.cuda.get_device_name(0))
""")
md("""## 2. Training data (Step 3)
The harness builder makes prompts exactly as the bot sees them; `agent.training.prepare_sft`
shortens the DOCUMENTS block of the 215 prompts longer than the fixed `max_seq_len` (2048) so the
answer is never truncated; the leakage check must say OK.""")
code("""
!python -m harness.build_sft_data --out data_sft.jsonl
!python -m agent.training.prepare_sft data_sft.jsonl data_sft_fit.jsonl
!python -m harness.leakage_check data_sft_fit.jsonl
!cp data_sft_fit.jsonl.report.json {DRIVE_DIR}/
""")
md("## 3. QLoRA SFT (A1) — skip if `adapters/a1` already exists on Drive")
code("""
import json, yaml, time, inspect, os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

cfg = yaml.safe_load(open("harness/config.yaml")); T = cfg["training"]; BASE = cfg["model"]["base_model"]; SEED = cfg["seeds"][0]
DONE = os.path.exists("adapters/a1/adapter_config.json")
print("adapter already trained:", DONE, "| budget:", T)

def make_args(cls, **kw):
    names = set(getattr(cls, "__dataclass_fields__", {})) or set(inspect.signature(cls).parameters)
    if "max_length" not in names and "max_seq_length" in names and "max_length" in kw:
        kw["max_seq_length"] = kw.pop("max_length")
    print("ignored by this TRL version:", [k for k in kw if k not in names])
    return cls(**{k: v for k, v in kw.items() if k in names})

if not DONE:
    rows = [json.loads(l) for l in open("data_sft_fit.jsonl")][: T["max_train_examples"]]
    ds = Dataset.from_list([{"prompt": r["messages"][:-1], "completion": [r["messages"][-1]]} for r in rows]).shuffle(seed=SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.config.use_cache = False
    lora = LoraConfig(r=T["lora_r"], lora_alpha=T["lora_alpha"], lora_dropout=T["lora_dropout"],
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM")
    args = make_args(SFTConfig, output_dir="outputs/sft", num_train_epochs=T["epochs"], learning_rate=T["learning_rate"],
        per_device_train_batch_size=1, gradient_accumulation_steps=8, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="steps", save_steps=100, save_total_limit=2, fp16=True, bf16=False,
        gradient_checkpointing=True, max_length=T["max_seq_len"], completion_only_loss=True, seed=SEED, report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    last = None
    if os.path.isdir("outputs/sft"):
        cks = sorted([d for d in os.listdir("outputs/sft") if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
        last = f"outputs/sft/{cks[-1]}" if cks else None
    print("resume from:", last)
    t0 = time.time(); trainer.train(resume_from_checkpoint=last); minutes = (time.time() - t0) / 60
    trainer.model.save_pretrained("adapters/a1"); tok.save_pretrained("adapters/a1")
    meta = {"gpu": torch.cuda.get_device_name(0), "train_minutes_this_session": round(minutes, 1),
            "resumed_from": last, "n_examples": len(rows), "settings": T, "seed": SEED,
            "log_history": trainer.state.log_history}
    json.dump(meta, open("adapters/a1/training_meta.json", "w"), indent=1)
    print(f"done in {minutes:.1f} min")
""")
code("""
import json, matplotlib.pyplot as plt
meta = json.load(open("adapters/a1/training_meta.json"))
h = [x for x in meta["log_history"] if "loss" in x]
plt.figure(figsize=(6,3.5)); plt.plot([x["step"] for x in h], [x["loss"] for x in h], color="#2a78d6", lw=2)
plt.xlabel("optimizer step"); plt.ylabel("training loss"); plt.title("G34 A1 QLoRA SFT loss"); plt.grid(alpha=.3)
plt.savefig("adapters/a1/sft_loss.png", dpi=160, bbox_inches="tight"); plt.show()
print(meta["gpu"], meta["train_minutes_this_session"], "min")
""")
code("""
# optional: upload the adapter to Hugging Face (create a WRITE token at huggingface.co/settings/tokens)
if HF_REPO:
    from huggingface_hub import login, HfApi
    login()
    HfApi().create_repo(HF_REPO, private=True, exist_ok=True)
    HfApi().upload_folder(folder_path="adapters/a1", repo_id=HF_REPO)
    print("https://huggingface.co/" + HF_REPO)
else:
    print("Adapter is on Drive:", DRIVE_DIR + "/adapters/a1  (share that folder link in MODEL_CARD.md)")
""")
md("""## 4. Experiments on the practice set (Step 5)
Free the GPU first (restart the runtime after training, then re-run part 1 only).
`g34_run.py` skips finished runs. To split work between members, give each a different `--configs`.""")
code("""
!python scripts/g34_run.py --split dev --adapter-path adapters/a1 --configs M0+A0,M1+A0,M0+A1,M1+A1
""")
code("""
!python -m harness.summarize --group G34 --split dev
!mkdir -p report && python -m harness.summarize --group G34 --split dev --latex > report/table_2x2_dev.tex
!python scripts/dashboard.py --group G34 --split dev
""")
md("""## 5. Memory ablation (Step 5.2): summary length 40 words vs the default 120
Kept under a separate group id (`G34abl40`) so the main results stay untouched.""")
code("""
!python scripts/g34_run.py --split dev --group G34abl40 --configs M1+A0 --env QA_M1_MAX_WORDS=40
!python -m harness.summarize --group G34abl40 --split dev
""")
md("## 6. Secret test (only inside the 48-hour Arena window)")
code("""
SECRET = "secret_test_questions.jsonl"   # upload it next to this notebook's repo folder
!python scripts/g34_run.py --split test --questions {SECRET} --adapter-path adapters/a1
!python -m harness.trace_export --group G34 --split test
!python scripts/dashboard.py --group G34 --split test
""")
md("## 7. Regression gate (real model), trace bundle, error candidates")
code("""
!python -m harness.smoke --group G34 --memory M1 --adapter A0 --accept
!python -m harness.trace_export --group G34
!python scripts/error_candidates.py --group G34 --split dev --n 80 > ERRORS_candidates.csv
!cp ERRORS_candidates.csv report/dashboard_* {DRIVE_DIR}/ 2>/dev/null; ls {DRIVE_DIR}
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "G34_colab_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
