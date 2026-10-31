"""Build the fixed GSM8K subset. Run once; the output is committed and never regenerated."""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED, N = 20261031, 300
out = Path(__file__).parent / "gsm8k_300.json"
assert not out.exists(), "data already built; do not regenerate"

ds = load_dataset("openai/gsm8k", "main", split="test")
ids = random.Random(SEED).sample(range(len(ds)), N)  # sample order == packing order
items = [{"pos": p, "test_idx": i, "question": ds[i]["question"], "answer": ds[i]["answer"]}
         for p, i in enumerate(ids)]
out.write_text(json.dumps({"seed": SEED, "n": N, "source": "openai/gsm8k main test", "items": items}, indent=1))
print("wrote", out, "first ids", ids[:5])
