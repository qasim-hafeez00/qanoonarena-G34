# MODEL_CARD.md — Group G34, training type A1 (QLoRA SFT)

> Fill the cells marked **TODO** from `adapters/a1/training_meta.json`, `results/*.json` and the
> notebook output (`notebooks/G34_colab_pipeline.ipynb`, parts 3–4).

## Adapter
| Item | Value |
| --- | --- |
| Download link (Hugging Face or Google Drive) | **TODO** |
| Base model | Qwen/Qwen3-4B-Instruct-2507 (from harness/config.yaml), loaded in 4-bit NF4 |
| Method | QLoRA SFT, completion-only loss (loss on the JSON answer only) |
| LoRA target modules | q, k, v, o, gate, up, down projections |
| Adapter hash (`adapter_hash` in `results/G34_M0+A1_dev_s13.json`) | **TODO** |

## Training data
| Item | Value |
| --- | --- |
| Source | `python -m harness.build_sft_data --out data_sft.jsonl` on the train split (M0 window as MEMORY, the bot's exact prompt) |
| Fitting to `max_seq_len` | `python -m agent.training.prepare_sft data_sft.jsonl data_sft_fit.jsonl`: 215 of 1851 examples (11.6%) were longer than 2048 Qwen tokens (max 2392); their DOCUMENTS block was shortened from the end so the answer is never truncated. After: 0 over the limit, max 2048 |
| Number of SFT examples | 1851 (all gold turns of the train split; under the 2000 limit) |
| Mix by type | T1 800, T2 358, T4 307, T3 199, T5 187 (one example per training session) |
| Leakage check passed | `OK: no overlap with evaluation questions.` (run on `data_sft_fit.jsonl`) |
| Never used for training | dev, smoke, secret test, any group's `contrib/` questions |

Reproduce: notebook part 2 (three commands). Everything is deterministic: the shuffle uses seed 13.

## Settings (match harness/config.yaml)
| Setting | Value |
| --- | --- |
| Epochs | 2 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate (SFT) | 2e-4, cosine schedule, 3% warm-up |
| Batch | 1 × gradient accumulation 8 (effective 8), about 463 optimizer steps |
| Max sequence length | 2048 |
| Precision | fp16 compute, 4-bit NF4 base with double quantisation, gradient checkpointing |
| Seed | 13 |

## Cost
| Item | Value |
| --- | --- |
| GPU | **TODO** (e.g. Tesla T4 16 GB, Colab) |
| Training time (wall clock) | **TODO** min |
| GPU-hours | **TODO** |

## Training curves
![SFT loss](adapters/a1/sft_loss.png) **TODO: copy the PNG into report/ and link it here**

## Results on the practice set (mean of 3 seeds)
| Version | Score | Accuracy | Citation F1 |
| --- | --- | --- | --- |
| M0+A0 | **TODO** | | |
| M0+A1 | **TODO** | | |

## Known failure cases
At least 3 examples where the trained model is wrong, with the error code (E1 to E8), taken from
`ERRORS.csv` rows with config `M0+A1` or `M1+A1`. **TODO**
