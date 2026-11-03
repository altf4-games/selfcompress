# Self-Compression in small quantized reasoning models

Controlled study of whether "Self-Compression" (shorter per-question reasoning when one prompt packs several
independent questions) appears in Qwen3-0.6B and Qwen3-1.7B, and whether it survives F16, Q8_0 and Q4_K_M GGUF
quantization. Runtime: llama.cpp b12019 (commit 1052f7e1d). Data: GSM8K test (300 questions, seed 20261031) and
MATH-500 levels 1-3 (100 questions, seed 20261101, plus 136 further questions, seed 20261102). Extended batch: k=8,
Qwen3-4B (Q4_K_M), DeepSeek-R1-Distill-Qwen-1.5B (F16), statement/empty/concise controls on Q4_K_M, a third seed.

## Layout
- `data/` fixed question lists and the scripts that make them
- `configs/` one YAML per run (model, GGUF, k, condition, seed, sampling); `configs/queue*.txt` are run queues
- `src/prompts.py` prompt builder and controls; `src/run.py` async, resumable llama.cpp client writing `runs/<id>.jsonl`
- `src/score.py` Math-Verify scoring, omission / cap-hit / loop detectors
- `src/make_tables.py`, `src/make_math_table.py` canonical analysis (group-level lengths, cluster bootstrap) -> `analysis/*.csv`
- `src/analyze.py` per-run summaries (not the canonical length statistics, see its docstring)
- `src/handcheck*.py`, `src/test_score.py` validity checks
- `runs/` raw generation logs (never edited); `scripts/` server, conversion and queue helpers

## Reproduce
1. Download llama.cpp b12019 and the Qwen3-0.6B / 1.7B Hugging Face checkpoints; convert and quantize with
   `scripts/convert.ps1` (F16 via `convert_hf_to_gguf.py`, then `llama-quantize` to Q8_0 and Q4_K_M).
2. `python scripts/make_configs.py` (and `make_extra.py`, `make_math_cfg.py`, `make_final_cfg.py`) write the configs and queues.
3. `python scripts/queue.py configs/queue.txt` starts a server per (model, precision) and runs every config; reruns resume.
4. Extended batch: `python scripts/download_extra.py` (Qwen3-4B, R1-Distill-Qwen-1.5B checkpoints), `scripts/convert_extra.ps1`,
   `python scripts/make_ext_cfg.py`, then `python scripts/queue.py configs/queue_x1.txt` (also `queue_x2.txt`, `queue_x3.txt`, `queue_fill2.txt`).
5. `python src/make_tables.py`, `src/make_math_table.py`, `src/extra_analysis.py` and `src/make_ext_tables.py` regenerate all tables and figures
   (`analysis/*.csv`; the k-sweep, extra-model and seed tables are in `analysis/results_ext.csv`).

## Notes
- Sampling is temperature 0.6, top_p 0.95, top_k 20, min_p 0, max 8192 new tokens, thinking mode on.
- Only an RTX 4060 Laptop GPU (8 GB) was used.
- Copyright (c) 2026 Pradyum Mistry and Dr. Smita Sankhe. All rights reserved; see `LICENSE`.
