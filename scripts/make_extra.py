"""Extra runs after the main grid. Writes configs and the queue files.
  python scripts/queue.py configs/queue2.txt   quantized 1.7B padding + 1.7B F16 seed 2
  python scripts/queue.py configs/queue3.txt   0.6B quantized padding, more seed-2 sweeps
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = {"q17": "Qwen3-1.7B", "q06": "Qwen3-0.6B"}
PREC = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_K_M"}
SERVER = {("q17", "f16"): (4, 20480), ("q17", "q8"): (6, 32768), ("q17", "q4"): (8, 40960)}
COND = {"k1": ("multi", 1), "k2": ("multi", 2), "k4": ("multi", 4), "pad": ("padding", 1)}
# (model, precision, arm, seed)
QUEUES = {
    "queue2": [("q17", "q8", "pad", 1), ("q17", "q4", "pad", 1), ("q17", "f16", "k1", 2),
               ("q17", "f16", "k4", 2), ("q17", "f16", "pad", 2)],
    "queue3": [("q06", "q8", "pad", 1), ("q06", "q4", "pad", 1),
               ("q06", "f16", "k1", 2), ("q06", "f16", "k4", 2), ("q06", "f16", "pad", 2),
               ("q17", "f16", "k2", 2),
               ("q17", "q4", "k1", 2), ("q17", "q4", "k4", 2), ("q17", "q4", "pad", 2),
               ("q17", "q8", "k1", 2), ("q17", "q8", "k4", 2), ("q17", "q8", "pad", 2)],
}
for qname, entries in QUEUES.items():
    lines = []
    for m, p, arm, seed in entries:
        cond, k = COND[arm]
        rid = f"{m}_{p}_{arm}" + (f"_s{seed}" if seed != 1 else "")
        slots, ctx = SERVER.get((m, p), (8, 40960))
        cfg = (f"run_id: {rid}\nmodel: {NAME[m]}\ngguf: models/{NAME[m]}-{PREC[p]}.gguf\nprecision: {PREC[p]}\n"
               f"condition: {cond}\nk: {k}\n" + ("k_ref: 4\n" if cond != "multi" else "") +
               f"seed: {seed}\nn_questions: 300\nserver: http://127.0.0.1:8089\nslots: {slots}\nmax_tokens: 8192\n"
               "sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n")
        path = ROOT / "configs" / f"{rid}.yaml"
        if not path.exists():
            path.write_text(cfg, encoding="utf-8")
        lines.append(f"{m} {p} {slots} {ctx} {rid}")
    (ROOT / "configs" / f"{qname}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(qname, len(lines), "runs")
