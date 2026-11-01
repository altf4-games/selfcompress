"""MATH-500 (levels 1-3, 100 questions) configs for Qwen3-1.7B F16: k=1, k=4 and padding, seed 1.
Writes configs/m500_*.yaml and configs/queue_math.txt. Run: python scripts/queue.py configs/queue_math.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lines = []
for arm, cond, k in (("k1", "multi", 1), ("k4", "multi", 4), ("pad", "padding", 1)):
    rid = f"m500_q17_f16_{arm}"
    cfg = (f"run_id: {rid}\nmodel: Qwen3-1.7B\ngguf: models/Qwen3-1.7B-F16.gguf\nprecision: F16\ndata_file: math500_L123_100.json\n"
           f"condition: {cond}\nk: {k}\n" + ("k_ref: 4\n" if cond != "multi" else "") +
           "seed: 1\nn_questions: 100\nserver: http://127.0.0.1:8089\nslots: 4\nmax_tokens: 8192\n"
           "sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n")
    (ROOT / "configs" / f"{rid}.yaml").write_text(cfg, encoding="utf-8")
    lines.append(f"q17 f16 4 20480 {rid}")
(ROOT / "configs" / "queue_math.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "runs queued")
