"""Final batch, run after queue_math: configs + configs/queue_final.txt (priority order).
  1. second seed for the 0.6B Q8_0 and Q4_K_M cells (k=1, k=4, padding)
  2. MATH-500 (levels 1-3, 100 q) on 0.6B F16 and on 1.7B Q4_K_M
  3. second seed of MATH-500 on 1.7B F16
Run: python scripts/queue.py configs/queue_final.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = {"q17": "Qwen3-1.7B", "q06": "Qwen3-0.6B"}
PREC = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_K_M"}
SERVER = {("q17", "f16"): (4, 20480), ("q17", "q8"): (6, 32768), ("q17", "q4"): (8, 40960)}
ARMS = {"k1": ("multi", 1), "k4": ("multi", 4), "pad": ("padding", 1)}
# (model, precision, dataset, seed)
BATCH = [("q06", "q8", "gsm", 2), ("q06", "q4", "gsm", 2),
         ("q06", "f16", "m500", 1), ("q17", "q4", "m500", 1), ("q17", "f16", "m500", 2)]
lines = []
for m, p, ds, seed in BATCH:
    slots, ctx = SERVER.get((m, p), (8, 40960))
    for arm, (cond, k) in ARMS.items():
        rid = ("m500_" if ds == "m500" else "") + f"{m}_{p}_{arm}" + (f"_s{seed}" if seed != 1 else "")
        n = 100 if ds == "m500" else 300
        cfg = (f"run_id: {rid}\nmodel: {NAME[m]}\ngguf: models/{NAME[m]}-{PREC[p]}.gguf\nprecision: {PREC[p]}\n"
               + ("data_file: math500_L123_100.json\n" if ds == "m500" else "") +
               f"condition: {cond}\nk: {k}\n" + ("k_ref: 4\n" if cond != "multi" else "") +
               f"seed: {seed}\nn_questions: {n}\nserver: http://127.0.0.1:8089\nslots: {slots}\nmax_tokens: 8192\n"
               "sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n")
        path = ROOT / "configs" / f"{rid}.yaml"
        if not path.exists():
            path.write_text(cfg, encoding="utf-8")
        lines.append(f"{m} {p} {slots} {ctx} {rid}")
(ROOT / "configs" / "queue_final.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "runs queued")
