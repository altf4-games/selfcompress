"""Extension of the MATH-500 subset: the remaining level 1-3 questions (not in math500_L123_100.json), fixed shuffled order, cut to a multiple of 4.
Run once; never regenerate. Seed 20261102."""
import json
import random
from pathlib import Path

from datasets import load_dataset

out = Path(__file__).parent / "math500_L123_ext.json"
assert not out.exists(), "data already built; do not regenerate"
used = {x["test_idx"] for x in json.loads((Path(__file__).parent / "math500_L123_100.json").read_text())["items"]}
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
rest = [i for i in range(len(ds)) if int(ds[i]["level"]) <= 3 and i not in used]
random.Random(20261102).shuffle(rest)
rest = rest[: len(rest) // 4 * 4]
items = [{"pos": p, "test_idx": i, "question": ds[i]["problem"], "gold": ds[i]["answer"], "level": int(ds[i]["level"])}
         for p, i in enumerate(rest)]
out.write_text(json.dumps({"seed": 20261102, "n": len(items), "source": "HuggingFaceH4/MATH-500 test, levels 1-3, not in the first 100", "items": items}, indent=1))
print("wrote", out, len(items))
