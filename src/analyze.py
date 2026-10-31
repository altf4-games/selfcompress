"""Tables and paired bootstrap CIs. Usage: python src/analyze.py runs/a.jsonl runs/b.jsonl ...

Per run: accuracy, omitted rate, tokens per question asked / answered / correct, cap-hit, loop, empty-think.
If a k=1 run and a k>1 run share a model+precision+seed, prints rho (median and mean) with a paired
bootstrap CI over question-matched tokens-per-question.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score_generation  # noqa: E402

B = 2000
rng = np.random.default_rng(0)


def load(path):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in recs:
        r["score"] = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
    return recs


def per_question_tokens(recs):
    """completion tokens per question asked, keyed by test_idx (each unit's tokens split evenly)."""
    d = {}
    for r in recs:
        for t in r["test_idx"]:
            d[t] = r["completion_tokens"] / r["k"]
    return d


def summarize(name, recs):
    sc = [r["score"] for r in recs]
    asked = sum(s["k"] for s in sc)
    ncorrect = sum(sum(s["correct"]) for s in sc)
    nomit = sum(sum(s["omitted"]) for s in sc)
    nanswered = asked - nomit
    toks = sum(r["completion_tokens"] for r in recs)
    ptok = sum(r["prompt_tokens"] for r in recs)
    q = np.array(list(per_question_tokens(recs).values()))
    return {
        "run": name, "n_units": len(recs), "q_asked": asked,
        "acc": ncorrect / asked, "omitted": nomit / asked,
        "tok/q_asked_mean": toks / asked, "tok/q_asked_median": float(np.median(q)),
        "tok/q_answered": toks / max(nanswered, 1), "tok/correct": toks / max(ncorrect, 1),
        "prompt_tok/q": ptok / asked,
        "cap_hit": np.mean([s["cap_hit"] for s in sc]), "loop": np.mean([s["loop"] for s in sc]),
        "empty_think": np.mean([s["empty_think"] for s in sc]),
        "think_frac": sum(r["think_tokens"] for r in recs) / max(toks, 1),
    }


def rho_ci(base, other):
    a, b = per_question_tokens(base), per_question_tokens(other)
    ids = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    out = {}
    for stat, f in (("median", np.median), ("mean", np.mean)):
        point = 1 - f(y) / f(x)
        bs = []
        for _ in range(B):
            ix = rng.integers(0, len(ids), len(ids))
            bs.append(1 - f(y[ix]) / f(x[ix]))
        out[stat] = (point, *np.percentile(bs, [2.5, 97.5]))
    return len(ids), out


if __name__ == "__main__":
    runs = {Path(p).stem: load(p) for p in sys.argv[1:]}
    for name, recs in runs.items():
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarize(name, recs).items()})
    names = list(runs)
    for n in names[1:]:
        n_ids, r = rho_ci(runs[names[0]], runs[n])
        print(f"rho {n} vs {names[0]} (n={n_ids} shared questions):",
              {s: tuple(round(float(v), 3) for v in t) for s, t in r.items()}, "(point, lo, hi)")
