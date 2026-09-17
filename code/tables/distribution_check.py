"""Distribution-level comparison applied to the same runs, at zero additional cost.

The reviewers asked what audit replication adds over distribution matching. The
honest way to answer is to apply a distribution-level criterion to the very same
simulation output and see whether it separates the models in the same way.

The audit reports the composition of what the platform recommends. The
distribution-level analogue available here is the composition of the slate: for
each model we compare the category distribution of the delivered slate in the late
rounds against the distribution a preference-free platform would deliver, which is
the frozen pool itself. Divergence from the pool is what a distribution-matching
study would call a departure from the reference.
"""
from __future__ import annotations
import os





import json
import math
import pathlib
import statistics as st
from collections import Counter

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "output"

MODELS = [
    ("Solar-Pro4 (22B)", "gate_runs"),
    ("Mistral-Small (24B)", "gate_runs_mistralai-mistral-small-24b-instruct-2501"),
    ("Llama-3.1 (8B)", "gate_runs_meta-llama-llama-3-1-8b-instruct"),
    ("Llama-3.2 (1B)", "gate_runs_meta-llama-llama-3-2-1b-instruct"),
]
CONDS = ["pref_neg", "pref_pos", "pref_h"]
LATE = (10, 11, 12)


def seeds_for(d):
    n = 0
    while all((DATA / d / f"{c}_s{n + 1}.json").exists() for c in CONDS) and n < 100:
        n += 1
    return range(1, n + 1)


def kl(p, q):
    return sum(p[k] * math.log(p[k] / q[k]) for k in p if p[k] > 0)


def main():
    pool = json.load(open(DATA / "pool.json", encoding="utf-8"))
    cats = sorted({it["category"] for it in pool["items"]})
    ref = {c: sum(1 for it in pool["items"] if it["category"] == c) / len(pool["items"])
           for c in cats}

    rows = []
    for name, d in MODELS:
        seeds = list(seeds_for(d))
        per_cond = {}
        for cond in CONDS:
            ds = []
            for s in seeds:
                log = json.load(open(DATA / d / f"{cond}_s{s}.json", encoding="utf-8"))
                cnt = Counter()
                for rd in log["rounds"]:
                    if rd["r"] in LATE:
                        for c in rd["pick_cat"]:
                            if c:
                                cnt[c] += 1
                tot = sum(cnt.values())
                if tot:
                    p = {c: (cnt[c] + 0.5) / (tot + 0.5 * len(cats)) for c in cats}
                    ds.append(kl(p, ref))
            per_cond[cond] = st.mean(ds) if ds else float("nan")
        rows.append((name, len(seeds), per_cond))

    print(f"{'model':24s} {'n':>3} {'KL neg':>8} {'KL pos':>8} {'KL h':>8} {'spread':>8}")
    for name, n, pc in rows:
        vals = [pc[c] for c in CONDS]
        print(f"{name:24s} {n:3d} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f} "
              f"{max(vals) - min(vals):8.3f}")

    with open(OUT / "t4_distribution.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Kullback--Leibler divergence from the frozen pool when a "
                  "distribution-level criterion is applied to the same runs.}\n")
        f.write("\\label{tab:distribution}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & $n$ & Negative pref. & Positive pref. & "
                  "Meaningless pref. \\\\\n\\midrule\n")
        for name, n, pc in rows:
            f.write(f"{name} & {n} & {pc['pref_neg']:.3f} & {pc['pref_pos']:.3f} & "
                    f"{pc['pref_h']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\nwrote {OUT}/t4_distribution.tex")


if __name__ == "__main__":
    main()
