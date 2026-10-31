"""Tables and cluster-bootstrap CIs.

Usage: python src/analyze.py base.jsonl other1.jsonl other2.jsonl ... [--ref ref.jsonl]

Every arm is compared on the SAME questions. The bootstrap resamples CLUSTERS of 4 consecutive questions
(the k=4 prompt groups; k=1/2 and the controls are built from the same groups) so that questions packed
into one prompt are not treated as independent, and the same resamples are used for all arms (paired).

Per-question tokens for k>1 are the prompt's completion tokens split evenly over its k questions.
Reported per arm: accuracy (+CI), omitted, tokens per question asked / answered / correct, cap-hit, loop,
empty-think. Versus the base arm: rho = 1 - L(arm)/L(base) (median and mean) and the accuracy difference.
With --ref: ratio L(arm)/L(ref) and accuracy difference versus the reference arm (e.g. the padding control).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score_generation  # noqa: E402

B, CLUSTER = 2000, 4
rng = np.random.default_rng(0)


def load(path):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in recs:
        r["score"] = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
    return recs


def per_question(recs):
    """question_pos -> (tokens per question asked, correct 0/1)."""
    d = {}
    for r in recs:
        for j, p in enumerate(r["question_pos"]):
            d[p] = (r["completion_tokens"] / r["k"], float(r["score"]["correct"][j]))
    return d


def summarize(name, recs):
    sc = [r["score"] for r in recs]
    asked = sum(s["k"] for s in sc)
    ncorrect = sum(sum(s["correct"]) for s in sc)
    nomit = sum(sum(s["omitted"]) for s in sc)
    toks = sum(r["completion_tokens"] for r in recs)
    ptok = sum(r["prompt_tokens"] for r in recs)
    q = np.array([v[0] for v in per_question(recs).values()])
    return {
        "run": name, "n_units": len(recs), "q_asked": asked,
        "acc": ncorrect / asked, "omitted": nomit / asked,
        "tok/q_asked_mean": toks / asked, "tok/q_asked_median": float(np.median(q)),
        "tok/q_answered": toks / max(asked - nomit, 1), "tok/correct": toks / max(ncorrect, 1),
        "prompt_tok/q": ptok / asked,
        "cap_hit": np.mean([s["cap_hit"] for s in sc]), "loop": np.mean([s["loop"] for s in sc]),
        "empty_think": np.mean([s["empty_think"] for s in sc]),
        "think_frac": sum(r["think_tokens"] for r in recs) / max(toks, 1),
    }


def ci(point, samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    args = sys.argv[1:]
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i:i + 2]
    paths = args + ([ref] if ref and ref not in args else [])
    runs = {Path(p).stem: load(p) for p in paths}
    names = [Path(p).stem for p in args]
    base = names[0]

    for n in runs:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarize(n, runs[n]).items()})

    pq = {n: per_question(r) for n, r in runs.items()}
    pos = sorted(set.intersection(*(set(d) for d in pq.values())))
    tok = {n: np.array([pq[n][p][0] for p in pos]) for n in runs}
    cor = {n: np.array([pq[n][p][1] for p in pos]) for n in runs}
    clusters = {}
    for i, p in enumerate(pos):
        clusters.setdefault(p // CLUSTER, []).append(i)
    clusters = [np.array(v) for v in clusters.values()]
    boots = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(B)]
    print(f"\n{len(pos)} shared questions in {len(clusters)} clusters; {B} cluster-bootstrap resamples (paired)")

    print("\naccuracy [95% CI]:")
    for n in runs:
        print(f"  {n}: {ci(cor[n].mean(), [cor[n][b].mean() for b in boots])}")

    def rel(a, b, f):  # L(a)/L(b)
        return lambda ix: f(tok[a][ix]) / f(tok[b][ix])

    print(f"\nrho = 1 - L(arm)/L({base})  (point [95% CI]):")
    for n in names[1:]:
        for stat, f in (("median", np.median), ("mean", np.mean)):
            fn = rel(n, base, f)
            print(f"  {n:<16} {stat:<6} rho {ci(1 - fn(slice(None)), [1 - fn(b) for b in boots])}")
        d = cor[n] - cor[base]
        print(f"  {n:<16} acc diff vs base {ci(d.mean(), [d[b].mean() for b in boots])}")

    if ref:
        r = Path(ref).stem
        print(f"\nratio L(arm)/L({r}) (<1 means shorter than the reference); acc diff vs {r}:")
        for n in names:
            if n == r:
                continue
            for stat, f in (("median", np.median), ("mean", np.mean)):
                fn = rel(n, r, f)
                print(f"  {n:<16} {stat:<6} ratio {ci(fn(slice(None)), [fn(b) for b in boots])}")
            d = cor[n] - cor[r]
            print(f"  {n:<16} acc diff vs ref {ci(d.mean(), [d[b].mean() for b in boots])}")


if __name__ == "__main__":
    main()
