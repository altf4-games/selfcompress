"""Build the fixed MATH-500 subset (levels 1-3, 100 questions). Run once; never regenerate.

Levels 4-5 are excluded on purpose: with an 8192-token cap and k=4 questions per prompt they would be
dominated by cap hits on a 1.7B model. Document this deviation in the paper.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED, N = 20261101, 100
out = Path(__file__).parent / "math500_L123_100.json"
assert not out.exists(), "data already built; do not regenerate"

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
pool = [i for i in range(len(ds)) if int(ds[i]["level"]) <= 3]
ids = random.Random(SEED).sample(pool, N)  # sample order == packing order
items = [{"pos": p, "test_idx": i, "question": ds[i]["problem"], "gold": ds[i]["answer"], "level": int(ds[i]["level"])}
         for p, i in enumerate(ids)]
out.write_text(json.dumps({"seed": SEED, "n": N, "source": "HuggingFaceH4/MATH-500 test, levels 1-3", "pool": len(pool),
                           "items": items}, indent=1))
print("wrote", out, "pool size", len(pool), "first ids", ids[:5])
