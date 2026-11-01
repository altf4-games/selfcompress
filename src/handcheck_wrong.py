"""List the generations in the hand-check sample (same fixed seed as handcheck.py) that were scored wrong or omitted.

Usage: python src/handcheck_wrong.py
Reading these is where a scoring or extraction bug would show up.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import extract_boxed, score_generation, split_think  # noqa: E402

FILES = ["q17_f16_k1", "q17_f16_k2", "q17_f16_k4", "q17_f16_pad", "q17_f16_stmt", "q17_f16_empty",
         "q06_f16_k1", "q06_f16_k2", "q06_f16_k4", "q17_q8_k1", "q17_q8_k2", "q17_q8_k4"]
gens = []
for f in FILES:
    for line in open(Path("runs") / f"{f}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        r["score"] = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
        gens.append(r)
random.Random(7).shuffle(gens)
sample = gens[:100]
bad = [r for r in sample if not all(r["score"]["correct"])]
print(f"sample of 100: fully correct {100 - len(bad)}, not fully correct {len(bad)}, "
      f"cap hits {sum(r['score']['cap_hit'] for r in sample)}")
for r in bad:
    _, ans, closed = split_think(r["raw_output"])
    boxed = extract_boxed(ans) if closed else []
    print(f"{r['run_id']} {r['unit_id']} gold={r['gold']} boxed={boxed} correct={r['score']['correct']} "
          f"finish={r['finish_reason']} tok={r['completion_tokens']}\n    tail: {r['raw_output'][-100:]!r}")
