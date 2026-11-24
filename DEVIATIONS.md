# DEVIATIONS.md — Group G34 (M1 summary memory × A1 QLoRA SFT)

**No value in `harness/config.yaml` was changed, and no file in `harness/` was edited.**
The model, corpus, retriever settings, prompt cap, memory budget, decoding settings, training limits
and the three seeds (13, 42, 2026) are the fixed ones. All submitted results use `--backend hf`.

The design choices below are **not** deviations. They are listed so the instructor can check them.

| Date | What we did | Why | Which results it affects |
| --- | --- | --- | --- |
| 2026-11-24 | SFT data: after `harness.build_sft_data`, `agent/training/prepare_sft.py` shortens the DOCUMENTS block of the 215/1851 examples longer than `max_seq_len` = 2048 Qwen tokens (lowest-ranked document first). The prompt template, memory, question and target answer are untouched. | Without it, TRL truncates on the right and cuts off the answer, which is the only part the loss is computed on. `max_seq_len` itself is unchanged. | A1 adapter, so M0+A1 and M1+A1 |
| 2026-11-24 | M1 summariser calls the same backbone with the LoRA adapter **disabled**, and with a shorter output cap (2 × target words + 40 tokens, at most the fixed 384). Same seed, temperature and top-p as the config. | The summariser is then identical in M1+A0 and M1+A1, and the answer-tuned adapter cannot turn a summary into a JSON answer. The cap only saves GPU time. | M1+A0, M1+A1 |
| 2026-11-24 | M1 summariser tokens are not counted in `tokens_in` / `tokens_out` of the results files, because the fixed evaluator only counts the answer call. We log them in every `memory_write` span (`tokens_in`, `tokens_out` of the `summary` item) and report them separately in the cost table. | Honest cost reporting. | Cost table only |
| 2026-11-24 | Memory ablation (Step 5.2) runs under the group id `G34abl40`, using the environment variable `QA_M1_MAX_WORDS=40`. | Keeps the main `G34_*` results files untouched. | Ablation table only |
| 2026-11-24 | Local development only: Ollama `qwen3:4b-instruct-2507-q4_K_M` (the same model, quantised) through `--backend openai`, to try the summary prompt. | No local CUDA GPU. These runs are **not** in `results/`. | None |
