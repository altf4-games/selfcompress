"""Extra analyses from existing logs (no new generations): cost per correct answer, position effects, self-checking markers.

Writes analysis/extra_results.csv. Same unit as make_tables.py: 75 groups of 4 consecutive questions, cluster bootstrap over groups.
Usage: python src/extra_analysis.py
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
B = 4000
GX = np.random.default_rng(11).integers(0, 75, (B, 75))
CELLS = ["q17_f16", "q17_q8", "q17_q4", "q06_f16", "q06_q8", "q06_q4"]
WAIT = re.compile(r"\bwait\b", re.I)
CHECK = re.compile(r"\b(double-check|double check|check again|let me check|let me verify|verify)\b", re.I)
out = []
eff = []


def log(*row):
    out.append(list(row))


def load(rid):
    """Per-group sums of completion tokens, prompt tokens, wait count, check count, plus per-question arrays."""
    cache = json.loads((ROOT / "analysis" / "cache" / f"{rid}.v2.json").read_text())
    cor, om = np.array(cache["cor"]), np.array(cache["om"])
    comp, prm, wait, chk, compnc = (np.zeros(75) for _ in range(5))
    for line in (ROOT / "runs" / f"{rid}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        g = r["question_pos"][0] // 4
        comp[g] += r["completion_tokens"]
        prm[g] += r["prompt_tokens"]
        if r["finish_reason"] != "length":  # capped generations excluded from marker counts (a few are runaway "Wait." loops)
            text = r["raw_output"]
            wait[g] += len(WAIT.findall(text))
            chk[g] += len(CHECK.findall(text))
            compnc[g] += r["completion_tokens"]
    return {"cor": cor, "om": om, "comp": comp, "compnc": compnc, "prm": prm, "wait": wait, "chk": chk,
            "corg": cor.reshape(75, 4).sum(axis=1)}


def ci(f):
    pt = f(np.arange(75))
    bs = np.array([f(x) for x in GX])
    return pt, *np.percentile(bs, [2.5, 97.5])


def fmt(t, d=2):
    return f"{t[0]:.{d}f} ({t[1]:.{d}f}, {t[2]:.{d}f})"


print("== tokens per correct answer (completion only; total incl. prompt)  and  self-checking per 1000 completion tokens")
for c in CELLS:
    d = {a: load(f"{c}_{a}") for a in ("k1", "k4", "pad")}
    row = [c]
    for a, v in d.items():
        tpc = ci(lambda X, v=v: v["comp"][X].sum() / v["corg"][X].sum())
        tot = ci(lambda X, v=v: (v["comp"][X].sum() + v["prm"][X].sum()) / v["corg"][X].sum())
        w = ci(lambda X, v=v: 1000 * v["wait"][X].sum() / v["compnc"][X].sum())
        k = ci(lambda X, v=v: 1000 * v["chk"][X].sum() / v["compnc"][X].sum())
        for lab, t in (("tok_per_correct", tpc), ("tok_incl_prompt_per_correct", tot), ("wait_per_1k", w), ("check_per_1k", k)):
            log(c, a, lab, *t)
        row.append(f"{a}: comp/correct {tpc[0]:.0f}, incl prompt {tot[0]:.0f}, wait/1k {w[0]:.1f}, check/1k {k[0]:.1f}")
    r41 = ci(lambda X: (d["k4"]["comp"][X].sum() / d["k4"]["corg"][X].sum()) / (d["k1"]["comp"][X].sum() / d["k1"]["corg"][X].sum()))
    log(c, "k4_over_k1", "tok_per_correct_ratio", *r41)
    wr = ci(lambda X: (d["k4"]["wait"][X].sum() / d["k4"]["compnc"][X].sum()) / (d["k1"]["wait"][X].sum() / d["k1"]["compnc"][X].sum()))
    log(c, "k4_over_k1", "wait_density_ratio", *wr)
    tpc = {a: ci(lambda X, a=a: d[a]["comp"][X].sum() / d[a]["corg"][X].sum()) for a in d}
    wd = {a: ci(lambda X, a=a: 1000 * d[a]["wait"][X].sum() / d[a]["compnc"][X].sum()) for a in ("k1", "k4")}
    LAB = {"q17": "1.7B", "q06": "0.6B", "f16": "F16", "q8": "Q8\\_0", "q4": "Q4\\_K\\_M"}
    eff.append([LAB[c[:3]], LAB[c[4:]], f"{tpc['k1'][0]:.0f}", f"{tpc['k4'][0]:.0f}", f"{tpc['pad'][0]:.0f}", fmt(r41),
                f"{wd['k1'][0]:.1f} / {wd['k4'][0]:.1f}", fmt(wr)])
    print(c, " | ".join(row[1:]))
    print("   k4/k1 tokens per correct:", fmt(r41), " wait density ratio k4/k1:", fmt(wr))

print("\n== position effects at k=4 (slot 1..4), seed 1 and 2 pooled over the six cells")
slot_om = np.zeros((4, 0))
for slot in range(4):
    oms, dacc = [], []
    for c in CELLS:
        for s in ("", "_s2"):
            if not (ROOT / "analysis" / "cache" / f"{c}_k4{s}.v2.json").exists():
                continue
            k4 = load(f"{c}_k4{s}")
            k1 = load(f"{c}_k1{s}")
            q = np.arange(slot, 300, 4)
            oms.append(k4["om"][q])
            dacc.append(k4["cor"][q] - k1["cor"][q])
    om = np.stack(oms)      # cells x 75 groups
    da = np.stack(dacc)
    pt_om = ci(lambda X: 100 * om[:, X].mean())
    pt_da = ci(lambda X: 100 * da[:, X].mean())
    log("pooled", f"slot{slot + 1}", "omitted_pct", *pt_om)
    log("pooled", f"slot{slot + 1}", "dacc_vs_k1_pts", *pt_da)
    print(f"slot {slot + 1}: omitted {fmt(pt_om, 1)} %   acc change vs k=1 {fmt(pt_da, 1)} pts")

# later slots (3,4) minus earlier slots (1,2): paired contrast of the accuracy change
def slot_da(slots):
    rows = []
    for c in CELLS:
        for s in ("", "_s2"):
            if not (ROOT / "analysis" / "cache" / f"{c}_k4{s}.v2.json").exists():
                continue
            k4, k1 = load(f"{c}_k4{s}"), load(f"{c}_k1{s}")
            rows.append(np.mean([(k4["cor"][np.arange(sl, 300, 4)] - k1["cor"][np.arange(sl, 300, 4)]) for sl in slots], axis=0))
    return np.stack(rows)


late, early = slot_da((2, 3)), slot_da((0, 1))
t = ci(lambda X: 100 * (late[:, X].mean() - early[:, X].mean()))
log("pooled", "late_minus_early", "dacc_pts", *t)
print("later (3,4) minus earlier (1,2) accuracy change:", fmt(t, 1), "pts")

tex = ["\\begin{table*}[!t]", "\\centering", "\\footnotesize",
       "\\caption{Cost per correct answer and backtracking density (GSM8K, seed 1). Tokens per correct answer = completion tokens of all "
       "generations in the arm divided by the number of correct answers. Ratio = $k{=}4$ over $k{=}1$ (95\\% CI). Wait density = occurrences of "
       "the word ``wait'' per 1000 completion tokens in generations that did not hit the cap, $k{=}1$ / $k{=}4$, and its $k{=}4$ over $k{=}1$ ratio.}",
       "\\label{tab:eff}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llccccc c}", "\\toprule",
       "Model & Prec. & Tokens/correct $k{=}1$ & $k{=}4$ & padding & Ratio $k{=}4$/$k{=}1$ & Wait/1k tokens $k{=}1$ / $k{=}4$ & Wait ratio \\\\",
       "\\midrule"]
tex += [" & ".join(r) + " \\\\" for r in eff]
tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
(ROOT / "paper" / "tables" / "tab_eff.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

with open(ROOT / "analysis" / "extra_results.csv", "w", newline="") as f:
    csv.writer(f).writerows([["cell", "arm", "key", "point", "lo", "hi"]] + out)
