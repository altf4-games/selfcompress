"""Extended batch (2026-11-01): configs + queue files.
  queue_x1.txt  k=8 sweep and k=8 padding (F16, first 288 questions)
  queue_x2.txt  Qwen3-4B Q4_K_M and R1-Distill-Qwen-1.5B F16 (k=1, k=4, padding)
  queue_x3.txt  statement/empty/concise controls on Q4_K_M, concise on F16, MATH-500 extension, third seed
Run: python scripts/queue.py configs/queue_x1.txt  (and so on)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = {"q17": "Qwen3-1.7B", "q06": "Qwen3-0.6B", "q4b": "Qwen3-4B", "r15": "R1-Distill-Qwen-1.5B"}
PREC = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_K_M"}
SAMPLING = "sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n"


def cfg(rid, m, p, cond, k, n, seed, slots, k_ref=None, data=None):
    s = (f"run_id: {rid}\nmodel: {NAME[m]}\ngguf: models/{NAME[m]}-{PREC[p]}.gguf\nprecision: {PREC[p]}\n"
         + (f"data_file: {data}\n" if data else "") + f"condition: {cond}\nk: {k}\n" + (f"k_ref: {k_ref}\n" if k_ref else "")
         + f"seed: {seed}\nn_questions: {n}\nserver: http://127.0.0.1:8089\nslots: {slots}\nmax_tokens: 8192\n" + SAMPLING)
    path = ROOT / "configs" / f"{rid}.yaml"
    if not path.exists():
        path.write_text(s, encoding="utf-8")


queues = {"x1": [], "x2": [], "x3": []}


def mk(q, m, p, slots, ctx, rid, cond, k, n, seed=1, k_ref=None, data=None):
    cfg(rid, m, p, cond, k, n, seed, slots, k_ref, data)
    queues[q].append(f"{m} {p} {slots} {ctx} {rid}")


# x1: k=8 (288 questions = 36 groups of 8 = 72 groups of 4)
mk("x1", "q06", "f16", 4, 40960, "q06_f16_k8", "multi", 8, 288)
mk("x1", "q06", "f16", 4, 40960, "q06_f16_pad8", "padding", 1, 288, k_ref=8)
mk("x1", "q17", "f16", 2, 20480, "q17_f16_k8", "multi", 8, 288)
mk("x1", "q17", "f16", 2, 20480, "q17_f16_pad8", "padding", 1, 288, k_ref=8)

# x2: new models (GSM8K, 300 questions)
for m, p, slots, ctx in (("q4b", "q4", 3, 24576), ("r15", "f16", 8, 40960)):
    for arm, cond, k, kr in (("k1", "multi", 1, None), ("k4", "multi", 4, None), ("pad", "padding", 1, 4)):
        mk("x2", m, p, slots, ctx, f"{m}_{p}_{arm}", cond, k, 300, k_ref=kr)

# x3: controls on quantized cells, concise on F16
for m, slots, ctx in (("q17", 8, 40960), ("q06", 8, 40960)):
    for cond, tag, kr in (("statement", "stmt", 4), ("empty", "empty", 4), ("concise", "conc", None)):
        mk("x3", m, "q4", slots, ctx, f"{m}_q4_{tag}", cond, 1, 300, k_ref=kr)
mk("x3", "q06", "f16", 8, 40960, "q06_f16_conc", "concise", 1, 300)
mk("x3", "q17", "f16", 4, 20480, "q17_f16_conc", "concise", 1, 300)
# MATH-500 extension (136 further level 1-3 questions)
D = "math500_L123_ext.json"
for arm, cond, k, kr in (("k1", "multi", 1, None), ("pad", "padding", 1, 4), ("k4", "multi", 4, None)):
    mk("x3", "q17", "q4", 4, 40960, f"m500x_q17_q4_{arm}", cond, k, 136, k_ref=kr, data=D)
for arm, cond, k, kr, slots in (("k1", "multi", 1, None, 4), ("pad", "padding", 1, 4, 4), ("k4", "multi", 4, None, 2)):
    mk("x3", "q17", "f16", slots, 20480, f"m500x_q17_f16_{arm}", cond, k, 136, k_ref=kr, data=D)
# third seed, F16 GSM8K
for m, slots, ctx in (("q17", 4, 20480), ("q06", 8, 40960)):
    for arm, cond, k, kr in (("k1", "multi", 1, None), ("k4", "multi", 4, None), ("pad", "padding", 1, 4)):
        mk("x3", m, "f16", slots, ctx, f"{m}_f16_{arm}_s3", cond, k, 300, seed=3, k_ref=kr)

for q, rows in queues.items():
    (ROOT / "configs" / f"queue_{q}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(q, len(rows), "runs")
