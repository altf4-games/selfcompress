"""Extra runs after the main grid: padding control on quantized 1.7B (RQ2) and a second seed on 1.7B F16.
Writes configs and configs/queue2.txt. Run with: python scripts/queue.py configs/queue2.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = {"f16": (4, 20480), "q8": (6, 32768), "q4": (8, 40960)}
PREC = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_K_M"}
# (precision, arm, condition, k, seed, suffix)
EXTRA = [("q8", "pad", "padding", 1, 1, ""), ("q4", "pad", "padding", 1, 1, ""),
         ("f16", "k1", "multi", 1, 2, "_s2"), ("f16", "k4", "multi", 4, 2, "_s2"), ("f16", "pad", "padding", 1, 2, "_s2")]
lines = []
for p, arm, cond, k, seed, suf in EXTRA:
    rid = f"q17_{p}_{arm}{suf}"
    slots, ctx = SERVER[p]
    cfg = (f"run_id: {rid}\nmodel: Qwen3-1.7B\ngguf: models/Qwen3-1.7B-{PREC[p]}.gguf\nprecision: {PREC[p]}\n"
           f"condition: {cond}\nk: {k}\n" + ("k_ref: 4\n" if cond != "multi" else "") +
           f"seed: {seed}\nn_questions: 300\nserver: http://127.0.0.1:8089\nslots: {slots}\nmax_tokens: 8192\n"
           "sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n")
    path = ROOT / "configs" / f"{rid}.yaml"
    if not path.exists():
        path.write_text(cfg, encoding="utf-8")
    lines.append(f"q17 {p} {slots} {ctx} {rid}")
(ROOT / "configs" / "queue2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "extra runs queued")
