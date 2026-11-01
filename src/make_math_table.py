"""MATH-500 (levels 1-3, 100 questions) results: paper/tables/tab_math.tex and analysis/results_math.csv.

Cells with all three runs (k=1, k=4, padding) complete are included; others are skipped.
Same definitions as make_tables.py: group-level length (mean tokens per question over 25 groups of 4), paired
cluster bootstrap over groups (B=4000), Math-Verify accuracy.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score_generation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N, G, B = 100, 25, 4000
GX = np.random.default_rng(2).integers(0, G, (B, G))
QX = (GX[:, :, None] * 4 + np.arange(4)).reshape(B, N)  # questions of the resampled groups
CELLS = [("1.7B", "F16", "m500_q17_f16", ""), ("1.7B", "F16 (seed 2)", "m500_q17_f16", "_s2"),
         ("1.7B", "Q4\\_K\\_M", "m500_q17_q4", ""), ("0.6B", "F16", "m500_q06_f16", "")]
EXPECT = {"k1": 100, "k4": 25, "pad": 100}
cache = {}


def complete(prefix, suf):
    for arm, n in EXPECT.items():
        f = ROOT / "runs" / f"{prefix}_{arm}{suf}.jsonl"
        if not f.exists() or sum(1 for _ in open(f, encoding="utf-8")) != n:
            return False
    return True


def run(rid):
    if rid not in cache:
        tok, cor, cap, om = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        for line in (ROOT / "runs" / f"{rid}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            s = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
            for j, p in enumerate(r["question_pos"]):
                tok[p], cor[p], cap[p], om[p] = r["completion_tokens"] / r["k"], s["correct"][j], s["cap_hit"], s["omitted"][j]
        cache[rid] = {"tok": tok, "cor": cor, "cap": cap, "om": om}
    return cache[rid]


def gtok(rid):
    return run(rid)["tok"].reshape(G, 4).mean(axis=1)


def ci(pt, bs):
    return pt, *np.percentile(bs, [2.5, 97.5])


def fmt(t, d=2, s=1):
    return f"{t[0] * s:.{d}f} ({t[1] * s:.{d}f}, {t[2] * s:.{d}f})"


def rho(a, b, f):
    return ci(1 - f(gtok(a)) / f(gtok(b)), 1 - f(gtok(a)[GX], axis=-1) / f(gtok(b)[GX], axis=-1))


def ratio(a, b, f):
    return ci(f(gtok(a)) / f(gtok(b)), f(gtok(a)[GX], axis=-1) / f(gtok(b)[GX], axis=-1))


def dacc(a, b):
    d = run(a)["cor"] - run(b)["cor"]
    return ci(d.mean(), d[QX].mean(axis=-1))


def acc(a):
    c = run(a)["cor"]
    return ci(c.mean(), c[QX].mean(axis=-1))


rows, out = [], []
for model, prec, prefix, suf in CELLS:
    if not complete(prefix, suf):
        print("skipping incomplete cell", prefix, suf)
        continue
    b, k4, pad = (f"{prefix}_{a}{suf}" for a in ("k1", "k4", "pad"))
    r = [model, prec, fmt(acc(b), 1, 100), fmt(acc(k4), 1, 100), fmt(dacc(k4, b), 1, 100)]
    for lab, t in (("rho_med_k4", rho(k4, b, np.median)), ("rho_mean_k4", rho(k4, b, np.mean)),
                   ("rho_med_pad", rho(pad, b, np.median)), ("rho_mean_pad", rho(pad, b, np.mean)),
                   ("ratio_med_k4_pad", ratio(k4, pad, np.median)), ("ratio_mean_k4_pad", ratio(k4, pad, np.mean)),
                   ("dacc_k4_pad", dacc(k4, pad))):
        out.append([prefix + suf, lab, *t])
        if lab != "dacc_k4_pad":
            r.append(fmt(t))
    r.append(f"{100 * run(b)['cap'].mean():.0f} / {100 * run(k4)['cap'].mean():.0f}")
    rows.append(r)
    for a in ("k1", "k4", "pad"):
        out.append([prefix + suf, f"om_{a}", 100 * run(f"{prefix}_{a}{suf}")["om"].mean(), 0, 0])
if rows:
    tex = ["\\begin{table*}[!t]", "\\centering", "\\footnotesize",
           "\\caption{MATH-500 (levels 1--3, 100 questions). $\\rho$ relative to $k{=}1$; ratio = $L$ of $k{=}4$ over the padding control; "
           "length statistics are group-of-four means.}", "\\label{tab:math}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llcccccccc}", "\\toprule",
           "Model & Prec. & Acc k=1 (\\%) & Acc k=4 (\\%) & $\\Delta$ acc (pts) & $\\rho_{med}$ k=4 & $\\rho_{mean}$ k=4 & "
           "$\\rho_{med}$ pad & $\\rho_{mean}$ pad & Cap hit k=1/k=4 (\\%) \\\\", "\\midrule"]
    merged = []
    for (model, prec, prefix, suf), r in zip([c for c in CELLS if complete(c[2], c[3])], rows):
        o = {x[1]: x for x in out if x[0] == prefix + suf}
        merged.append(r[:5] + [r[6], r[8], r[10], fmt(o["dacc_k4_pad"][2:], 1, 100), f"{o['om_k4'][2]:.0f}", r[-1]])
    tex[tex.index("\\begin{tabular}{llcccccccc}")] = "\\begin{tabular}{llcccccccccc}"
    tex[-2] = ("Model & Prec. & Acc k=1 (\\%) & Acc k=4 (\\%) & $\\Delta$ acc (pts) & $\\rho_{mean}$ k=4 & $\\rho_{mean}$ pad & "
               "$L_{mean}$ k=4 / pad & $\\Delta$ acc k=4 $-$ pad & Omitted k=4 (\\%) & Cap hit k=1/k=4 (\\%) \\\\")
    tex += [" & ".join(r) + " \\\\" for r in merged]
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (ROOT / "paper" / "tables" / "tab_math.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
if rows:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [c for c in CELLS if complete(c[2], c[3])]
    lab = [f"{m}\n{p.replace(chr(92), '').replace(' (seed 2)', ' s2')}" for m, p, _, _ in cells]
    names = [pre + suf for _, _, pre, suf in cells]
    x = np.arange(len(cells))
    get = lambda cell, key: next(r for r in out if r[0] == cell and r[1] == key)[2:]  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for off, key, col, nm in ((-0.19, "rho_mean_k4", "#0072B2", "k=4"), (0.19, "rho_mean_pad", "#E69F00", "padding")):
        ts = [get(c, key) for c in names]
        ax[0].bar(x + off, [t[0] for t in ts], 0.36, color=col, label=nm,
                  yerr=[[t[0] - t[1] for t in ts], [t[2] - t[0] for t in ts]], error_kw={"lw": 0.8, "capsize": 1.5})
    ax[0].set_title("(a) mean $\\rho$ vs $k{=}1$", fontsize=8); ax[0].legend(fontsize=6, frameon=False)
    ts1 = [dacc(f"{pre}_k4{suf}", f"{pre}_k1{suf}") for _, _, pre, suf in cells]
    ts2 = [get(c, "dacc_k4_pad") for c in names]
    for off, ts, col, nm in ((-0.19, ts1, "#0072B2", "k=4 vs k=1"), (0.19, ts2, "#E69F00", "k=4 vs padding")):
        ax[1].bar(x + off, [100 * t[0] for t in ts], 0.36, color=col, label=nm,
                  yerr=[[100 * (t[0] - t[1]) for t in ts], [100 * (t[2] - t[0]) for t in ts]], error_kw={"lw": 0.8, "capsize": 1.5})
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("(b) accuracy difference (pts)", fontsize=8)
    ax[1].legend(fontsize=6, frameon=False)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(lab, fontsize=5.5); a.tick_params(axis="y", labelsize=6)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figures" / "fig_math.pdf")
with open(ROOT / "analysis" / "results_math.csv", "w", newline="") as f:
    csv.writer(f).writerows([["cell", "key", "point", "lo", "hi"]] + out)
print(f"wrote MATH-500 tables for {len(rows)} cells")
