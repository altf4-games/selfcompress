"""Cross-cell results: LaTeX tables (paper/tables), CSV (analysis/), figure (paper/figures).

Usage: python src/make_tables.py
All CIs: 95% percentile, cluster bootstrap over 75 prompt groups of 4 consecutive questions, B=4000,
resamples shared across every arm (paired). Scored runs are cached in analysis/cache/.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score_generation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "analysis" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
(ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)
B = 4000
IDX = np.arange(300).reshape(75, 4)[np.random.default_rng(0).integers(0, 75, (B, 75))].reshape(B, 300)
ALL = np.arange(300)
CELLS = [("1.7B", "F16", "q17_f16"), ("1.7B", "Q8\\_0", "q17_q8"), ("1.7B", "Q4\\_K\\_M", "q17_q4"),
         ("0.6B", "F16", "q06_f16"), ("0.6B", "Q8\\_0", "q06_q8"), ("0.6B", "Q4\\_K\\_M", "q06_q4")]
RUNS = {}


def run(rid):
    if rid in RUNS:
        return RUNS[rid]
    f, c = ROOT / "runs" / f"{rid}.jsonl", CACHE / f"{rid}.v2.json"  # v2: tail-based loop detector
    size = f.stat().st_size
    if c.exists() and json.loads(c.read_text())["size"] == size:
        d = json.loads(c.read_text())
    else:
        tok, cor, cap, om = [0.0] * 300, [0.0] * 300, [0.0] * 300, [0.0] * 300
        meta = {"loop": [], "repeat": [], "empty_think": [], "think": 0, "comp": 0, "prompt": 0}
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            s = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
            for j, p in enumerate(r["question_pos"]):
                tok[p], cor[p], cap[p], om[p] = r["completion_tokens"] / r["k"], float(s["correct"][j]), float(s["cap_hit"]), float(s["omitted"][j])
            meta["loop"].append(s["loop"]); meta["repeat"].append(s["repeat"]); meta["empty_think"].append(s["empty_think"])
            meta["think"] += r["think_tokens"]; meta["comp"] += r["completion_tokens"]; meta["prompt"] += r["prompt_tokens"]
        d = {"size": size, "tok": tok, "cor": cor, "cap": cap, "om": om, "meta": meta}
        c.write_text(json.dumps(d))
    d = {k: (np.array(v) if k in ("tok", "cor", "cap", "om") else v) for k, v in d.items()}
    RUNS[rid] = d
    return d


def est(g):
    """g(X) -> statistic given index array X; returns (point, lo, hi)."""
    pt, bs = g(ALL), g(IDX)
    return pt, *np.percentile(bs, [2.5, 97.5])


# Length statistics are computed at the GROUP level for every arm: the mean tokens per question over each group of 4
# consecutive questions (75 groups). For k>1 a prompt's tokens are shared by its questions, so per-question values are
# prompt averages; comparing a median of averages with a median of single-question lengths would be biased.
# The mean over groups equals the mean over questions.
GX = np.random.default_rng(1).integers(0, 75, (B, 75))
GALL = np.arange(75)


def gtok(rid):
    return run(rid)["tok"].reshape(75, 4).mean(axis=1)


def est_g(g):
    pt, bs = g(GALL), g(GX)
    return pt, *np.percentile(bs, [2.5, 97.5])


med = lambda a, X: np.median(a[X], axis=-1)  # noqa: E731
mean = lambda a, X: np.mean(a[X], axis=-1)  # noqa: E731


def rho(a, b, f):  # 1 - L(a)/L(b), group level
    return est_g(lambda X: 1 - f(gtok(a), X) / f(gtok(b), X))


def ratio(a, b, f):  # L(a)/L(b), group level
    return est_g(lambda X: f(gtok(a), X) / f(gtok(b), X))


def dacc(a, b):  # acc(a) - acc(b)
    return est(lambda X: mean(run(a)["cor"], X) - mean(run(b)["cor"], X))


def acc(a):
    return est(lambda X: mean(run(a)["cor"], X))


def fmt(t, d=2, pct=False):
    s = 100 if pct else 1
    return f"{t[0] * s:.{d}f} ({t[1] * s:.{d}f}, {t[2] * s:.{d}f})"


def write_tex(name, header, rows, caption, label, star=True):
    env = "table*" if star else "table"
    cols = "ll" + "c" * (len(header) - 2)
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\resizebox{{\\{'textwidth' if star else 'columnwidth'}}}{{!}}{{%",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    (ROOT / "paper" / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


csv_rows = []
def log(table, cell, key, t):
    csv_rows.append([table, cell, key, f"{t[0]:.4f}", f"{t[1]:.4f}", f"{t[2]:.4f}"])


# ---- Table I: length compression rho vs k=1
rows = []
for m, p, c in CELLS:
    r = [m, p, f"{np.median(gtok(c + '_k1')):.0f}"]
    for f, fn in ((med, "median"), (mean, "mean")):
        for k in (2, 4):
            t = rho(f"{c}_k{k}", f"{c}_k1", f)
            r.append(fmt(t)); log("rho", c, f"{fn}_k{k}", t)
    rows.append(r)
write_tex("tab_rho", ["Model", "Prec.", "Median L(1)", "$\\rho_{med}$ k=2", "$\\rho_{med}$ k=4", "$\\rho_{mean}$ k=2", "$\\rho_{mean}$ k=4"],
          rows, "Compression rate $\\rho=1-L(k)/L(1)$ (95\\% cluster-bootstrap CI), seed 1, 300 GSM8K questions. $L$ = completion tokens per question asked.", "tab:rho")

# ---- Table II: accuracy and failure modes
rows = []
for m, p, c in CELLS:
    r = [m, p] + [fmt(acc(f"{c}_k{k}"), 1, True) for k in (1, 4)]
    t = dacc(f"{c}_k4", f"{c}_k1"); r.append(fmt(t, 1, True)); log("acc", c, "dacc_k4_k1", t)
    r += [f"{100 * run(c + '_k4')['om'].mean():.1f}", f"{100 * run(c + '_k1')['cap'].mean():.1f} / {100 * run(c + '_k4')['cap'].mean():.1f}"]
    rows.append(r)
write_tex("tab_acc", ["Model", "Prec.", "Acc k=1 (\\%)", "Acc k=4 (\\%)", "$\\Delta$ acc (pts)", "Omitted k=4 (\\%)", "Cap hit k=1/k=4 (\\%)"],
          rows, "Accuracy (Math-Verify) and failure modes. Omitted = question with no boxed answer; cap hit = 8192-token limit reached.", "tab:acc")

# ---- Table III: padding control comparison
rows = []
for m, p, c in CELLS:
    r = [m, p]
    for f, fn in ((med, "median"), (mean, "mean")):
        t = rho(f"{c}_pad", f"{c}_k1", f); r.append(fmt(t)); log("pad", c, f"rho_pad_{fn}", t)
    for f, fn in ((med, "median"), (mean, "mean")):
        t = ratio(f"{c}_k4", f"{c}_pad", f); r.append(fmt(t)); log("pad", c, f"ratio_k4_pad_{fn}", t)
    t = dacc(f"{c}_k4", f"{c}_pad"); r.append(fmt(t, 1, True)); log("pad", c, "dacc_k4_pad", t)
    rows.append(r)
pad_rows = rows

# ---- Table IV: other controls (F16 only)
rows = []
for m, c in (("1.7B", "q17_f16"), ("0.6B", "q06_f16")):
    r = [m, "F16"]
    for a in ("stmt", "empty", "pad"):
        for f, fn in ((med, "median"), (mean, "mean")):
            t = rho(f"{c}_{a}", f"{c}_k1", f); r.append(fmt(t)); log("ctrl", c, f"rho_{a}_{fn}", t)
    rows.append(r)
write_tex("tab_ctrl", ["Model", "Prec.", "stmt med", "stmt mean", "empty med", "empty mean", "pad med", "pad mean"],
          rows, "$\\rho$ of the three non-multi-question controls (F16): statement, empty-question and padding.", "tab:ctrl")

# ---- Table V: seed replication
rows = []
def has_s2(c):  # a cell is included only if its seed-2 runs are complete (k=1, k=4, padding)
    return all((ROOT / "runs" / f"{c}_{a}_s2.jsonl").exists() and
               sum(1 for _ in open(ROOT / "runs" / f"{c}_{a}_s2.jsonl", encoding="utf-8")) == (75 if a == "k4" else 300)
               for a in ("k1", "k4", "pad"))


for m, p, c in [x for x in CELLS if has_s2(x[2])]:
    r = [m, p]
    for s in ("", "_s2"):
        for f, fn in ((med, "median"), (mean, "mean")):
            t = rho(f"{c}_k4{s}", f"{c}_k1{s}", f); r.append(fmt(t)); log("seed", c, f"rho_k4_{fn}{s or '_s1'}", t)
    t = rho(f"{c}_k1_s2", f"{c}_k1", mean); r.append(fmt(t)); log("seed", c, "k1_seed_noise_mean", t)
    rows.append(r)
write_tex("tab_seed", ["Model", "Prec.", "$\\rho_{med}$ s1", "$\\rho_{mean}$ s1", "$\\rho_{med}$ s2", "$\\rho_{mean}$ s2", "seed noise (k=1 s2 vs s1)"],
          rows, "Replication with a second seed ($\\rho$ for k=4). Last column: $1-L_{s2}(1)/L_{s1}(1)$ for k=1, the length difference due to sampling alone.", "tab:seed")

# ---- Table VI: failure and structure diagnostics
rows = []
for m, p, c in CELLS:
    d1, d4 = run(f"{c}_k1")["meta"], run(f"{c}_k4")["meta"]
    pair = lambda key, f=1: f"{100 * np.mean(d1[key]) * f:.1f} / {100 * np.mean(d4[key]):.1f}"  # noqa: E731
    share = lambda d: f"{100 * d['think'] / d['comp']:.0f}"  # noqa: E731
    rows.append([m, p, pair("loop"), pair("repeat"), pair("empty_think"), f"{share(d1)} / {share(d4)}"])
write_tex("tab_diag", ["Model", "Prec.", "Loop (\%)", "Repeat (\%)", "Empty think (\%)", "Think share (\%)"],
          rows, "Diagnostics per generation, $k{=}1$ / $k{=}4$: loop = cap hit and a 10-gram repeated at least 50 times in the last 1500 words; repeat = repeated 10-gram only; think share = thinking tokens over completion tokens.", "tab:diag", star=False)

# ---- Table VII: tail test. rho at the median and the 90th percentile of per-question length
def rho_gq(a, b, q):
    return est_g(lambda X: 1 - np.quantile(gtok(a)[X], q, axis=-1) / np.quantile(gtok(b)[X], q, axis=-1))


def ratio_gq(a, b, q):
    return est_g(lambda X: np.quantile(gtok(a)[X], q, axis=-1) / np.quantile(gtok(b)[X], q, axis=-1))


rows = []
for m, p, c in CELLS:
    r = [m, p]
    for arm in ("k4", "pad"):
        for q in (0.5, 0.9):
            t = rho_gq(f"{c}_{arm}", f"{c}_k1", q); r.append(fmt(t)); log("tail", c, f"rho_{arm}_p{int(q * 100)}", t)
    for q in (0.5, 0.9):
        log("tail", c, f"ratio_k4_pad_p{int(q * 100)}", ratio_gq(f"{c}_k4", f"{c}_pad", q))
    rows.append(r)
rows = [pr[:4] + [tr[3], tr[5]] + pr[4:] for pr, tr in zip(pad_rows, rows)]
write_tex("tab_pad", ["Model", "Prec.", "$\\rho_{med}$ pad", "$\\rho_{mean}$ pad", "$\\rho_{p90}$ k=4", "$\\rho_{p90}$ pad",
                      "$L_{med}$ k=4 / pad", "$L_{mean}$ k=4 / pad", "$\\Delta$ acc k=4 $-$ pad"],
          rows, "Padding control ($k{=}1$ plus filler matched to the $k{=}4$ prompt length) and tail test. $\\rho_{p90}$ is the reduction at the 90th percentile of group-level length. A ratio below 1 means the $k{=}4$ prompt yields shorter reasoning than padding alone.", "tab:pad")

# ---- Figure 2: complementary CDF of per-question length (log x), tail shown directly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#0072B2", "#E69F00"
fig2, axs = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True, sharey=True)
for a, (m, p, c) in zip(axs.ravel(), CELLS):
    for key, col, ls, lab in (("k1", "k", "-", "k=1"), ("k4", BLUE, "-", "k=4"), ("pad", ORANGE, "--", "padding")):
        v = np.sort(gtok(f"{c}_{key}"))
        a.plot(v, 1 - np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.2, label=lab)
    a.set_xscale("log"); a.set_title(f"{m} {p.replace(chr(92), '')}", fontsize=7)
    a.tick_params(labelsize=6); a.spines[["top", "right"]].set_visible(False)
axs[0, 0].legend(fontsize=6, frameon=False)
for a in axs[1]:
    a.set_xlabel("mean tokens per question, per group of 4 (log)", fontsize=6)
for a in axs[:, 0]:
    a.set_ylabel("fraction of groups longer", fontsize=7)
fig2.tight_layout()
fig2.savefig(ROOT / "paper" / "figures" / "fig_dist.pdf"); fig2.savefig(ROOT / "paper" / "figures" / "fig_dist.png", dpi=200)

# ---- text claims: cap-hit sensitivity (1.7B F16) and 0.6B empty-control per-question medians
arms = [f"q17_f16_{a}" for a in ("k1", "k2", "k4", "pad")]
capg = np.any([run(a)["cap"].reshape(75, 4).max(axis=1) > 0 for a in arms], axis=0)
kept = np.where(~capg)[0]
kb = kept[np.random.default_rng(3).integers(0, len(kept), (B, len(kept)))]
r4, rp = gtok("q17_f16_k4"), gtok("q17_f16_pad")
log("sens", "q17_f16", "groups_dropped_capped", (float(capg.sum()),) * 3)
log("sens", "q17_f16", "ratio_mean_k4_pad_nocap",
    (r4[kept].mean() / rp[kept].mean(), *np.percentile(r4[kb].mean(axis=1) / rp[kb].mean(axis=1), [2.5, 97.5])))
for a in ("k1", "empty"):
    v = float(np.median(run(f"q06_f16_{a}")["tok"]))
    log("emptymed", "q06_f16", f"median_tokens_{a}", (v, v, v))

with open(ROOT / "analysis" / "results.csv", "w", newline="") as f:
    csv.writer(f).writerows([["table", "cell", "key", "point", "lo", "hi"]] + csv_rows)

# ---- Figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#6B6B6B"
labels = [f"{m}\n{p.replace(chr(92), '')}" for m, p, _ in CELLS]
x = np.arange(len(CELLS))
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for a, f, title in ((ax[0], med, "(a) median $\\rho$"), (ax[1], mean, "(b) mean $\\rho$")):
    for off, key, col, lab in ((-0.19, "k4", BLUE, "k=4"), (0.19, "pad", ORANGE, "padding")):
        ts = [rho(f"{c}_{key}", f"{c}_k1", f) for _, _, c in CELLS]
        a.bar(x + off, [t[0] for t in ts], 0.36, color=col, label=lab,
              yerr=[[t[0] - t[1] for t in ts], [t[2] - t[0] for t in ts]], error_kw={"lw": 0.8, "capsize": 1.5})
    a.set_title(title, fontsize=8); a.set_xticks(x); a.set_xticklabels(labels, fontsize=5.5); a.set_ylim(0, 0.8)
ax[0].set_ylabel("reduction in tokens per question", fontsize=7); ax[0].legend(fontsize=6, frameon=False)
for off, k, col in ((-0.19, 2, GREY), (0.19, 4, BLUE)):
    ts = [dacc(f"{c}_k{k}", f"{c}_k1") for _, _, c in CELLS]
    ax[2].bar(x + off, [100 * t[0] for t in ts], 0.36, color=col, label=f"k={k}",
              yerr=[[100 * (t[0] - t[1]) for t in ts], [100 * (t[2] - t[0]) for t in ts]], error_kw={"lw": 0.8, "capsize": 1.5})
ax[2].axhline(0, color="k", lw=0.6); ax[2].set_title("(c) accuracy change vs k=1 (pts)", fontsize=8)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=5.5); ax[2].legend(fontsize=6, frameon=False)
for a in ax:
    a.tick_params(axis="y", labelsize=6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "paper" / "figures" / "fig_main.pdf"); fig.savefig(ROOT / "paper" / "figures" / "fig_main.png", dpi=200)
print("wrote tables, results.csv, figure")
