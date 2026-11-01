"""Extended batch (2026-11-01): k-sweep to k=8, extra models, extra controls, third seed, pooled MATH-500.

Writes paper/tables/tab_ext_*.tex, paper/figures/fig_k.pdf and analysis/results_ext.csv.
Only complete runs are used; sections whose runs are missing are skipped. Same conventions as make_tables.py:
group-level length (mean tokens per question over a group of consecutive questions), paired cluster bootstrap over
groups (B=4000, resamples shared across arms), Math-Verify accuracy.
Usage: python src/make_ext_tables.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from score import score_generation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "analysis" / "cache"
B = 4000
rows_csv = []
RUNS = {}


def cfg(rid):
    return yaml.safe_load((ROOT / "configs" / f"{rid}.yaml").read_text(encoding="utf-8-sig"))


def complete(rid):
    f = ROOT / "runs" / f"{rid}.jsonl"
    if not f.exists():
        return False
    c = cfg(rid)
    need = c["n_questions"] // c["k"] if c["condition"] == "multi" else c["n_questions"]
    return sum(1 for _ in open(f, encoding="utf-8")) == need


def run(rid):
    """Per-question arrays tok (completion tokens / k), cor, cap, om, indexed by question position."""
    if rid in RUNS:
        return RUNS[rid]
    f, c = ROOT / "runs" / f"{rid}.jsonl", CACHE / f"{rid}.ext.json"
    n = cfg(rid)["n_questions"]
    if c.exists() and json.loads(c.read_text())["size"] == f.stat().st_size:
        d = json.loads(c.read_text())
    else:
        d = {"size": f.stat().st_size, "tok": [0.0] * n, "cor": [0.0] * n, "cap": [0.0] * n, "om": [0.0] * n}
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            s = score_generation(r["raw_output"], r["gold"], r["finish_reason"])
            for j, p in enumerate(r["question_pos"]):
                d["tok"][p], d["cor"][p] = r["completion_tokens"] / r["k"], float(s["correct"][j])
                d["cap"][p], d["om"][p] = float(s["cap_hit"]), float(s["omitted"][j])
        c.write_text(json.dumps(d))
    RUNS[rid] = {k: np.array(v) for k, v in d.items() if k != "size"}
    return RUNS[rid]


class Frame:
    """A set of arms evaluated on the first n questions with resampling units of U consecutive questions."""

    def __init__(self, n, U):
        self.n, self.U = n, U
        self.G = n // U
        self.GX = np.random.default_rng(5).integers(0, self.G, (B, self.G))
        self.QX = (self.GX[:, :, None] * U + np.arange(U)).reshape(B, n)

    def arr(self, rid, key):
        return rid[key][: self.n] if isinstance(rid, dict) else run(rid)[key][: self.n]

    def gtok(self, rid):
        return self.arr(rid, "tok").reshape(self.G, self.U).mean(axis=1)

    @staticmethod
    def ci(pt, bs):
        return pt, *np.percentile(bs, [2.5, 97.5])

    def rho(self, a, b, f=np.mean):
        ga, gb = self.gtok(a), self.gtok(b)
        return self.ci(1 - f(ga) / f(gb), 1 - f(ga[self.GX], axis=-1) / f(gb[self.GX], axis=-1))

    def ratio(self, a, b, f=np.mean):
        ga, gb = self.gtok(a), self.gtok(b)
        return self.ci(f(ga) / f(gb), f(ga[self.GX], axis=-1) / f(gb[self.GX], axis=-1))

    def dacc(self, a, b):
        d = self.arr(a, "cor") - self.arr(b, "cor")
        return self.ci(d.mean(), d[self.QX].mean(axis=-1))

    def acc(self, a):
        c = self.arr(a, "cor")
        return self.ci(c.mean(), c[self.QX].mean(axis=-1))

    def rate(self, a, key):
        return 100 * self.arr(a, key).mean()


def fmt(t, d=2, s=1):
    return f"{t[0] * s:.{d}f} ({t[1] * s:.{d}f}, {t[2] * s:.{d}f})"


def log(cell, key, t):
    rows_csv.append([cell, key, *(f"{x:.4f}" for x in t)])


def write_tex(name, header, rows, caption, label, star=True):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\resizebox{{\\{'textwidth' if star else 'columnwidth'}}}{{!}}{{%",
             f"\\begin{{tabular}}{{ll{'c' * (len(header) - 2)}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    (ROOT / "paper" / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def have(*rids):
    return all(complete(r) for r in rids)


# ---- 1. k-sweep to k=8 (F16, first 288 questions, units of 8)
sweep = []
for m, c in (("1.7B", "q17_f16"), ("0.6B", "q06_f16")):
    if not have(f"{c}_k1", f"{c}_k2", f"{c}_k4", f"{c}_k8", f"{c}_pad", f"{c}_pad8"):
        print("k-sweep incomplete for", c)
        continue
    F = Frame(288, 8)
    k1 = f"{c}_k1"
    r = [m, "F16"]
    for k in (2, 4, 8):
        t = F.rho(f"{c}_k{k}", k1); r.append(fmt(t)); log(c, f"sweep_rho_mean_k{k}", t)
    for a in ("pad", "pad8"):
        t = F.rho(f"{c}_{a}", k1); r.append(fmt(t)); log(c, f"sweep_rho_mean_{a}", t)
    for k in (4, 8):
        t = F.dacc(f"{c}_k{k}", k1); r.append(fmt(t, 1, 100)); log(c, f"sweep_dacc_k{k}", t)
    r += [f"{F.rate(f'{c}_k{k}', 'om'):.1f}" for k in (4, 8)]
    r.append(f"{F.rate(f'{c}_k8', 'cap'):.1f}")
    sweep.append((m, c, F, r))
if sweep:
    write_tex("tab_ext_k", ["Model", "Prec.", "$\\rho$ k=2", "$\\rho$ k=4", "$\\rho$ k=8", "$\\rho$ pad4", "$\\rho$ pad8",
                            "$\\Delta$ acc k=4", "$\\Delta$ acc k=8", "Omit. k=4", "Omit. k=8", "Cap k=8"],
              [s[3] for s in sweep],
              "Dose response to $k{=}8$ (F16, first 288 questions, resampling units of 8). Mean $\\rho$ relative to $k{=}1$; pad4/pad8 = padding "
              "control matched to the $k{=}4$/$k{=}8$ prompt. Accuracy change in points, omitted-question and cap-hit rates in \\%.", "tab:ext_k")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for (m, c, F, _), col in zip(sweep, ("#0072B2", "#E69F00")):
        ks = [1, 2, 4, 8]
        pts = [0.0] + [F.rho(f"{c}_k{k}", f"{c}_k1") for k in (2, 4, 8)]
        ax[0].errorbar(ks, [p if isinstance(p, float) else p[0] for p in pts],
                       yerr=[[0] + [p[0] - p[1] for p in pts[1:]], [0] + [p[2] - p[0] for p in pts[1:]]], color=col, marker="o", ms=3,
                       lw=1, capsize=2, label=f"{m} multi-question")
        pad = [F.rho(f"{c}_{a}", f"{c}_k1") for a in ("pad", "pad8")]
        ax[0].plot([4, 8], [p[0] for p in pad], color=col, ls="--", marker="s", ms=3, lw=1, label=f"{m} padding")
        om = [0.0] + [F.rate(f"{c}_k{k}", "om") for k in (2, 4, 8)]
        ax[1].plot(ks, om, color=col, marker="o", ms=3, lw=1, label=m)
    ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2)
    for a in ax:
        a.set_xticks([1, 2, 4, 8]); a.set_xticklabels(["1", "2", "4", "8"], fontsize=6); a.tick_params(axis="y", labelsize=6)
        a.set_xlabel("questions per prompt k", fontsize=7); a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title("(a) mean $\\rho$ vs $k{=}1$", fontsize=8); ax[0].legend(fontsize=5.5, frameon=False)
    ax[1].set_title("(b) omitted questions (%)", fontsize=8); ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(ROOT / "paper" / "figures" / "fig_k.pdf")

# ---- 2. extra models (GSM8K, 300 questions)
rows = []
for m, p, c in (("Qwen3-4B", "Q4\\_K\\_M", "q4b_q4"), ("R1-Distill-1.5B", "F16", "r15_f16"), ("Qwen3-1.7B", "F16", "q17_f16")):
    if not have(f"{c}_k1", f"{c}_k4", f"{c}_pad"):
        print("model cell incomplete:", c)
        continue
    F = Frame(300, 4)
    k1, k4, pad = f"{c}_k1", f"{c}_k4", f"{c}_pad"
    r = [m, p, fmt(F.acc(k1), 1, 100), fmt(F.dacc(k4, k1), 1, 100)]
    for lab, t in (("rho_med_k4", F.rho(k4, k1, np.median)), ("rho_mean_k4", F.rho(k4, k1)), ("rho_mean_pad", F.rho(pad, k1)),
                   ("ratio_mean_k4_pad", F.ratio(k4, pad)), ("dacc_k4_pad", F.dacc(k4, pad))):
        log(c, "model_" + lab, t)
        r.append(fmt(t, 1, 100) if lab == "dacc_k4_pad" else fmt(t))
    r += [f"{F.rate(k4, 'om'):.1f}", f"{F.rate(k1, 'cap'):.1f} / {F.rate(k4, 'cap'):.1f}", f"{np.median(F.gtok(k1)):.0f}"]
    rows.append(r)
if rows:
    write_tex("tab_ext_models", ["Model", "Prec.", "Acc k=1 (\\%)", "$\\Delta$ acc (pts)", "$\\rho_{med}$ k=4", "$\\rho_{mean}$ k=4",
                                 "$\\rho_{mean}$ pad", "$L_{mean}$ k=4 / pad", "$\\Delta$ acc k=4 $-$ pad", "Omit. k=4 (\\%)",
                                 "Cap k=1/k=4 (\\%)", "Median L(1)"], rows,
              "Additional models (GSM8K, 300 questions, seed 1): same measurements as the main grid. Qwen3-1.7B F16 is repeated for reference.", "tab:ext_models")

# ---- 3. additional controls: statement, empty, concise, padding vs k=4 (rho mean / median)
rows = []
for m, p, c in (("1.7B", "F16", "q17_f16"), ("1.7B", "Q4\\_K\\_M", "q17_q4"), ("0.6B", "F16", "q06_f16"), ("0.6B", "Q4\\_K\\_M", "q06_q4")):
    F = Frame(300, 4)
    r = [m, p]
    ok = False
    for a in ("k4", "pad", "stmt", "empty", "conc"):
        if have(f"{c}_{a}", f"{c}_k1"):
            ok = True
            t = F.rho(f"{c}_{a}", f"{c}_k1"); r.append(fmt(t)); log(c, f"ctrl_rho_mean_{a}", t)
        else:
            r.append("--")
    if ok:
        rows.append(r)
if rows:
    write_tex("tab_ext_ctrl", ["Model", "Prec.", "k=4", "padding", "statement", "empty", "concise"], rows,
              "Mean $\\rho$ relative to $k{=}1$ for $k{=}4$ and each single-question control (GSM8K, seed 1). Concise = added line ``Keep your reasoning "
              "concise.'' (our wording).", "tab:ext_ctrl", star=False)

# ---- 4. third seed, F16 GSM8K
rows = []
for m, c in (("1.7B", "q17_f16"), ("0.6B", "q06_f16")):
    if not have(*(f"{c}_{a}{s}" for a in ("k1", "k4", "pad") for s in ("", "_s2", "_s3"))):
        print("seed 3 incomplete for", c)
        continue
    F = Frame(300, 4)
    r = [m, "F16"]
    for s in ("", "_s2", "_s3"):
        t = F.rho(f"{c}_k4{s}", f"{c}_k1{s}"); r.append(fmt(t)); log(c, f"seed_rho_mean_k4{s or '_s1'}", t)
    # pooled over seeds: average per-question arrays across the three seeds
    pool = lambda a: {k: np.mean([run(f"{c}_{a}{s}")[k] for s in ("", "_s2", "_s3")], axis=0) for k in ("tok", "cor", "cap", "om")}  # noqa: E731
    k1, k4, pad = pool("k1"), pool("k4"), pool("pad")
    t = F.rho(k4, k1); r.append(fmt(t)); log(c, "seed_rho_mean_k4_pooled", t)
    t = F.rho(pad, k1); r.append(fmt(t)); log(c, "seed_rho_mean_pad_pooled", t)
    t = F.dacc(k4, k1); r.append(fmt(t, 1, 100)); log(c, "seed_dacc_k4_pooled", t)
    rows.append(r)
if rows:
    write_tex("tab_ext_seed3", ["Model", "Prec.", "$\\rho_{mean}$ s1", "s2", "s3", "pooled k=4", "pooled pad", "pooled $\\Delta$ acc (pts)"], rows,
              "Three seeds (F16, GSM8K). Pooled = per-question values averaged over the three seeds before the bootstrap.", "tab:ext_seed3", star=False)

# ---- 5. MATH-500 levels 1-3, pooled 100 + 136 questions (1.7B)
rows = []
for p, a, b in (("F16", "m500_q17_f16", "m500x_q17_f16"), ("Q4\\_K\\_M", "m500_q17_q4", "m500x_q17_q4")):
    if not have(*(f"{x}_{arm}" for x in (a, b) for arm in ("k1", "k4", "pad"))):
        print("MATH extension incomplete for", p)
        continue
    cat = lambda arm: {k: np.concatenate([run(f"{a}_{arm}")[k], run(f"{b}_{arm}")[k]]) for k in ("tok", "cor", "cap", "om")}  # noqa: E731
    k1, k4, pad = cat("k1"), cat("k4"), cat("pad")
    F = Frame(236, 4)
    r = ["1.7B", p, fmt(F.acc(k1), 1, 100), fmt(F.dacc(k4, k1), 1, 100)]
    for lab, t in (("rho_mean_k4", F.rho(k4, k1)), ("rho_mean_pad", F.rho(pad, k1)), ("ratio_mean_k4_pad", F.ratio(k4, pad)),
                   ("dacc_k4_pad", F.dacc(k4, pad))):
        log("math236_" + a[-3:], lab, t)
        r.append(fmt(t, 1, 100) if lab == "dacc_k4_pad" else fmt(t))
    r += [f"{F.rate(k4, 'om'):.1f}", f"{F.rate(k1, 'cap'):.1f} / {F.rate(k4, 'cap'):.1f}"]
    rows.append(r)
if rows:
    write_tex("tab_ext_math", ["Model", "Prec.", "Acc k=1 (\\%)", "$\\Delta$ acc (pts)", "$\\rho_{mean}$ k=4", "$\\rho_{mean}$ pad",
                               "$L_{mean}$ k=4 / pad", "$\\Delta$ acc k=4 $-$ pad", "Omit. k=4 (\\%)", "Cap k=1/k=4 (\\%)"], rows,
              "MATH-500 levels 1--3, all 236 questions (the first 100 plus 136 further ones), seed 1; 59 groups of four.", "tab:ext_math")

with open(ROOT / "analysis" / "results_ext.csv", "w", newline="") as f:
    csv.writer(f).writerows([["cell", "key", "point", "lo", "hi"]] + rows_csv)
print("wrote extended tables:", len(rows_csv), "statistics")
