"""Validity check #1: look for parsing/scoring bugs without reading every trace.

Usage: python src/handcheck.py runs/a.jsonl runs/b.jsonl ... [--n 100] [--out analysis/handcheck_sample.txt]

1. Cross-check: re-grade every non-omitted answer with a second, simple numeric comparison and list the
   generations where it disagrees with Math-Verify.
2. List loop-flagged generations (with the repeated 10-gram) and omissions that are NOT cap hits.
3. Write a fixed-seed random sample of N traces (head, tail, answer, grades) to the output file for reading.
The second grader shares the boxed-extraction step, so extraction bugs are caught only by reading the sample.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import extract_boxed, score_generation, split_think  # noqa: E402


def simple_num(s):
    s = re.sub(r"\\text\{[^}]*\}|\\(?:,|!|;| )|[$%]|\\%", "", s.replace("\\$", "")).replace(",", "")
    if "\\frac" in s or "/" in s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def main():
    args = sys.argv[1:]
    n, out = 100, "analysis/handcheck_sample.txt"
    for flag in ("--n", "--out"):
        if flag in args:
            i = args.index(flag)
            if flag == "--n":
                n = int(args[i + 1])
            else:
                out = args[i + 1]
            del args[i:i + 2]
    gens = []
    for p in args:
        for l in Path(p).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                r["score"] = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
                gens.append(r)
    print(f"{len(gens)} generations from {len(args)} files")

    dis, loops, omit_nocap = [], [], []
    for r in gens:
        s = r["score"]
        _, ans, closed = split_think(r["raw_output"])
        boxed = extract_boxed(ans) if closed else []
        used = boxed[-s["k"]:] if len(boxed) > s["k"] else boxed
        for i, g in enumerate(r["gold"]):
            if i < len(used):
                a, b = simple_num(used[i]), float(g)
                simple_ok = a is not None and abs(a - b) < 1e-6
                if a is not None and simple_ok != s["correct"][i]:
                    dis.append((r["run_id"], r["unit_id"], i, g, used[i], s["correct"][i]))
        if s["repeat"]:
            w = r["raw_output"].split()
            top = Counter(tuple(w[j:j + 10]) for j in range(len(w) - 9)).most_common(1)[0]
            loops.append((r["run_id"], r["unit_id"], r["finish_reason"], top[1], " ".join(top[0])[:80]))
        if any(s["omitted"]) and not s["cap_hit"]:
            omit_nocap.append((r["run_id"], r["unit_id"], r["finish_reason"], s["n_boxed"], s["k"], ans[-120:].replace("\n", " ")))

    print(f"\n[1] Math-Verify vs simple grader disagreements: {len(dis)}")
    for d in dis[:25]:
        print("   ", d)
    print(f"\n[2] loop-flagged generations: {len(loops)}  (finish reason / repeats / repeated 10-gram)")
    for l in loops[:25]:
        print("   ", l)
    print(f"\n[3] omissions that are NOT cap hits: {len(omit_nocap)}  (run, unit, finish, boxed, k, tail)")
    for o in omit_nocap[:25]:
        print("   ", o)

    random.Random(7).shuffle(gens)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in gens[:n]:
            think, ans, closed = split_think(r["raw_output"])
            f.write(f"=== {r['run_id']} {r['unit_id']} gold={r['gold']} finish={r['finish_reason']} "
                    f"tokens={r['completion_tokens']} score={r['score']['correct']} omitted={r['score']['omitted']}\n"
                    f"THINK HEAD: {think[:250]!r}\nTHINK TAIL: {think[-250:]!r}\nANSWER: {ans[:400]!r}\n\n")
    print(f"\nwrote {min(n, len(gens))} sampled traces to {out}")


if __name__ == "__main__":
    main()
