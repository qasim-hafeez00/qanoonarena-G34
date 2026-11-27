"""Step 3 (A1): QLoRA SFT of the fixed backbone on data_sft_fit.jsonl. Resumable.

    python -m agent.training.train_sft --data data_sft_fit.jsonl --out adapters/a1 --work outputs/sft

All budget values come from harness/config.yaml (epochs, LoRA r/alpha/dropout, learning rate,
max_seq_len, max_train_examples, seed). Loss is computed on the answer only (completion-only).
A checkpoint is written every 100 optimizer steps; a new run resumes from the latest one
(optimizer, scheduler and RNG state included), so a lost Colab runtime costs < 100 steps.

Each session appends its wall-clock time to <work>/sessions.json (updated at every logging step,
so a session that is killed still counts), which the model card sums into GPU-hours.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.common import load_config  # noqa: E402


def make_args(cls, **kw):
    names = set(getattr(cls, "__dataclass_fields__", {})) or set(inspect.signature(cls).parameters)
    if "max_length" not in names and "max_seq_length" in names and "max_length" in kw:
        kw["max_seq_length"] = kw.pop("max_length")
    print("ignored by this TRL version:", [k for k in kw if k not in names], flush=True)
    return cls(**{k: v for k, v in kw.items() if k in names})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data_sft_fit.jsonl")
    p.add_argument("--out", default="adapters/a1")
    p.add_argument("--work", default="outputs/sft")
    a = p.parse_args(argv)

    out, work = ROOT / a.out, ROOT / a.work
    if (out / "adapter_config.json").exists():
        print("adapter already trained:", out)
        return
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    cfg = load_config()
    T, base, seed = cfg["training"], cfg["model"]["base_model"], cfg["seeds"][0]
    print("budget:", T, flush=True)
    rows = [json.loads(l) for l in open(ROOT / a.data, encoding="utf-8")][: T["max_train_examples"]]
    ds = Dataset.from_list([{"prompt": r["messages"][:-1], "completion": [r["messages"][-1]]} for r in rows]).shuffle(seed=seed)

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model.config.use_cache = False
    lora = LoraConfig(r=T["lora_r"], lora_alpha=T["lora_alpha"], lora_dropout=T["lora_dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    args = make_args(SFTConfig, output_dir=str(work), num_train_epochs=T["epochs"], learning_rate=T["learning_rate"],
                     per_device_train_batch_size=1, gradient_accumulation_steps=8, lr_scheduler_type="cosine",
                     warmup_ratio=0.03, logging_steps=10, save_strategy="steps", save_steps=100, save_total_limit=2,
                     fp16=True, bf16=False, gradient_checkpointing=True, max_length=T["max_seq_len"],
                     completion_only_loss=True, seed=seed, report_to="none")

    work.mkdir(parents=True, exist_ok=True)
    sess_fp = work / "sessions.json"
    sessions = json.loads(sess_fp.read_text()) if sess_fp.exists() else []
    t0 = time.time()
    cks = sorted(work.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1]))
    last = str(cks[-1]) if cks else None
    me = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": torch.cuda.get_device_name(0),
          "resumed_from": last, "minutes": 0.0, "last_step": None}
    sessions.append(me)

    class Clock(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            me["minutes"] = round((time.time() - t0) / 60, 1)
            me["last_step"] = state.global_step
            sess_fp.write_text(json.dumps(sessions, indent=1))
            # keep every loss line, also across sessions
            with open(work / "loss_log.jsonl", "a") as f:
                f.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora,
                         callbacks=[Clock()])
    # Qwen3 ships bf16 weights, so PEFT creates the LoRA matrices in bf16; the T4 has no bf16 and the
    # fp16 GradScaler cannot unscale bf16 grads. Keep the trainable LoRA weights in fp32.
    for _, prm in trainer.model.named_parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    print("trainable params:", sum(x.numel() for x in trainer.model.parameters() if x.requires_grad), flush=True)
    print("resume from:", last, flush=True)
    trainer.train(resume_from_checkpoint=last)

    me["minutes"] = round((time.time() - t0) / 60, 1)
    sess_fp.write_text(json.dumps(sessions, indent=1))
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    total = sum(s["minutes"] for s in sessions)
    meta = {"gpu": me["gpu"], "sessions": sessions, "train_minutes_total": round(total, 1),
            "gpu_hours": round(total / 60, 2), "n_examples": len(rows), "settings": T, "seed": seed,
            "log_history": trainer.state.log_history}
    (out / "training_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {total:.1f} GPU-minutes over {len(sessions)} session(s)", flush=True)


if __name__ == "__main__":
    main()
