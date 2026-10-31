"""Generate the main-grid configs and configs/queue.txt (priority order). Existing configs are kept."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"q17": "Qwen3-1.7B", "q06": "Qwen3-0.6B"}
PREC = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_K_M"}
# (slots, shared KV pool tokens) per model/precision; chosen to fit 8 GB VRAM.
SERVER = {("q17", "f16"): (4, 20480), ("q17", "q8"): (6, 32768), ("q17", "q4"): (8, 40960),
          ("q06", "f16"): (8, 40960), ("q06", "q8"): (8, 40960), ("q06", "q4"): (8, 40960)}
ARMS = {"k1": ("multi", 1), "k2": ("multi", 2), "k4": ("multi", 4),
        "pad": ("padding", 1), "stmt": ("statement", 1), "empty": ("empty", 1)}
# Priority: 1.7B F16 incl. controls first, then the k-sweeps, controls for 0.6B last.
ORDER = [("q17", "f16", ["k1", "k4", "k2", "pad", "stmt", "empty"]), ("q06", "f16", ["k1", "k4", "k2"]),
         ("q17", "q8", ["k1", "k4", "k2"]), ("q17", "q4", ["k1", "k4", "k2"]),
         ("q06", "q8", ["k1", "k4", "k2"]), ("q06", "q4", ["k1", "k4", "k2"]),
         ("q06", "f16", ["pad", "stmt", "empty"])]

lines = []
for m, p, arms in ORDER:
    slots, ctx = SERVER[(m, p)]
    for a in arms:
        cond, k = ARMS[a]
        rid = f"{m}_{p}_{a}"
        cfg = (f"run_id: {rid}\nmodel: {MODELS[m]}\ngguf: models/{MODELS[m]}-{PREC[p]}.gguf\nprecision: {PREC[p]}\n"
               f"condition: {cond}\nk: {k}\n" + ("k_ref: 4\n" if cond != "multi" else "") +
               f"seed: 1\nn_questions: 300\nserver: http://127.0.0.1:8089\nslots: {slots}\nmax_tokens: 8192\n"
               "sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n")
        path = ROOT / "configs" / f"{rid}.yaml"
        if not path.exists():
            path.write_text(cfg, encoding="utf-8")
        lines.append(f"{m} {p} {slots} {ctx} {rid}")
(ROOT / "configs" / "queue.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "runs queued")
