"""Sanity tests for scoring. Run: python src/test_score.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import score_generation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def ok(raw, gold):
    return score_generation("<think>x</think>" + raw, gold, "stop")["correct"]


# MATH-style LaTeX golds and answers
assert ok("\\boxed{\\frac{1}{2}}", ["\\frac{1}{2}"]) == [True]
assert ok("\\boxed{\\frac{1}{2}}", ["\\frac12"]) == [True]
assert ok("\\boxed{3}", ["\\frac{1}{2}"]) == [False]
assert ok("\\boxed{(1,2)}", ["(1,2)"]) == [True]
# GSM8K-style plain numbers are unchanged
assert ok("\\boxed{18}", ["18"]) == [True] and ok("\\boxed{19}", ["18"]) == [False]
# Regression: the first 1.7B F16 k=1 run must still score 90.3% (value reported in the paper)
recs = [json.loads(l) for l in open(ROOT / "runs" / "q17_f16_k1.jsonl", encoding="utf-8")]
acc = sum(score_generation(r["raw_output"], r["gold"], r["finish_reason"])["correct"][0] for r in recs) / len(recs)
assert abs(acc - 0.9033) < 0.002, acc
print("all scoring tests passed; q17_f16_k1 accuracy", round(acc, 4))
