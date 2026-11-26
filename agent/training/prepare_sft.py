"""Step 3 data preparation for A1 (Group G34). Repeatable, no randomness except the fixed seed.

    python -m harness.build_sft_data --out data_sft.jsonl          # 1. harness builder (fixed)
    python -m agent.training.prepare_sft data_sft.jsonl data_sft_fit.jsonl   # 2. this script
    python -m harness.leakage_check data_sft_fit.jsonl              # 3. leakage check

Why this step exists: the harness builds prompts with the fixed 3072-token prompt cap, but
the fixed training budget is max_seq_len = 2048. Measured with the Qwen3 tokenizer, 215 of
1851 examples (11.6%) are longer than 2048 tokens. TRL truncates on the RIGHT, which would
cut off the assistant answer, i.e. the only part the loss is computed on.

What it does, per example (nothing else is changed):
  - if prompt + answer fits in max_seq_len tokens: keep it unchanged;
  - otherwise shorten the DOCUMENTS block from its END (the lowest-ranked retrieved
    document first, then a trimmed piece of the next one), exactly as the agent's own
    _build_messages trims documents when the prompt cap is reached, until it fits.
The system prompt, MEMORY, QUESTION and the target answer are never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.common import load_config  # noqa: E402


def fit_example(row: dict, tok, max_len: int, margin: int = 8) -> tuple[dict, int, int]:
    msgs = row["messages"]

    def n_tokens(m):
        ids = tok.apply_chat_template(m, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):  # transformers >= 5 returns a BatchEncoding
            ids = ids["input_ids"]
        return len(ids)

    before = n_tokens(msgs)
    if before <= max_len:
        return row, before, before
    user = msgs[1]["content"]
    head, sep, rest = user.partition("DOCUMENTS:\n")
    docs, sep2, tail = rest.partition("\n\nMEMORY:\n")
    if not sep or not sep2:
        return row, before, before  # unexpected layout: leave as is (reported)
    blocks = docs.split("\n\n[")
    blocks = [blocks[0]] + ["[" + b for b in blocks[1:]]
    over = before - max_len + margin
    # drop whole blocks from the end while that is not enough, then trim characters
    while len(blocks) > 1:
        last = blocks[-1]
        t = len(tok(last, add_special_tokens=False)["input_ids"])
        if t <= over:
            blocks.pop()
            over -= t
        else:
            break
    new_docs = "\n\n".join(blocks)
    new_msgs = None
    for _ in range(40):
        cut = max(0, len(new_docs) - over * 4)
        new_docs = new_docs[:cut]
        new_user = head + sep + (new_docs if new_docs.strip() else "(none found)") + sep2 + tail
        new_msgs = [msgs[0], dict(msgs[1], content=new_user)] + msgs[2:]
        after = n_tokens(new_msgs)
        if after <= max_len:
            break
        over = after - max_len + margin
    out = dict(row, messages=new_msgs)
    return out, before, after


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inp")
    p.add_argument("out")
    a = p.parse_args(argv)
    cfg = load_config()
    max_len = cfg["training"]["max_seq_len"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    n_trim, still_long, stats = 0, 0, []
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            r2, b, af = fit_example(r, tok, max_len)
            n_trim += b != af
            still_long += af > max_len
            stats.append((b, af))
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    report = {"examples": len(rows), "trimmed": n_trim, "still_over_limit": still_long,
              "max_tokens_before": max(b for b, _ in stats), "max_tokens_after": max(x for _, x in stats),
              "max_seq_len": max_len}
    Path(a.out + ".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
