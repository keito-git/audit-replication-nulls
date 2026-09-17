"""Stability of the verdict against the seed sample.

Defined in the framework lock before the confirmatory seeds were taken.

A single front-half / back-half split is one arbitrary partition. Repeating the
split many times turns the question "did the two halves agree?" into "how often
do two halves agree?", which is what the framework's direction-stability rule
needs. The cumulative curve answers the companion question of when the estimate
settles, and it is the figure that shows most directly that ten seeds were not
enough.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st
import sys

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, MODELS, seeds_for, norm_reinf  # noqa: E402

SPLITS = 1000
AGREE_THRESHOLD = 0.80          # frozen in the framework lock
ROPE = 0.05                     # smallest asymmetry called meaningful


def diffs(d, seeds):
    neg = [norm_reinf(json.load(open(DATA / d / f"pref_neg_s{s}.json",
                                     encoding="utf-8")))[0] for s in seeds]
    pos = [norm_reinf(json.load(open(DATA / d / f"pref_pos_s{s}.json",
                                     encoding="utf-8")))[0] for s in seeds]
    return [a - b for a, b in zip(neg, pos)]


def repeated_split(dd, rng):
    n = len(dd)
    h = n // 2
    agree = both = 0
    for _ in range(SPLITS):
        idx = list(range(n))
        rng.shuffle(idx)
        a = st.mean(dd[i] for i in idx[:h])
        b = st.mean(dd[i] for i in idx[h:])
        agree += (a > 0) == (b > 0)
        both += (a > 0) and (b > 0)
    return agree / SPLITS, both / SPLITS


def boot_ci(dd, rng, iters=4000):
    n = len(dd)
    ms = sorted(st.mean(dd[rng.randrange(n)] for _ in range(n)) for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters)]


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'model':22s} {'n':>4} {'mean':>9} {'agree':>7} {'both+':>7} "
          f"{'95% CI':>24} {'ROPE':>10}")
    for name, d in MODELS:
        seeds = list(seeds_for(d))
        if len(seeds) < 10:
            continue
        dd = diffs(d, seeds)
        rng = random.Random(20260914)
        agree, both = repeated_split(dd, rng)
        lo, hi = boot_ci(dd, rng)
        if hi < ROPE and lo > -ROPE:
            rope = "Excluded"
        elif lo > ROPE or hi < -ROPE:
            rope = "Above the bound"
        else:
            rope = "Inconclusive"
        rows.append({"model": name, "n": len(seeds), "mean": st.mean(dd),
                     "agree": agree, "both_pos": both, "ci": [lo, hi], "rope": rope})
        print(f"{name:22s} {len(seeds):4d} {st.mean(dd):+9.4f} {agree:7.3f} "
              f"{both:7.3f} [{lo:+.4f}, {hi:+.4f}] {rope:>10}")

    with open(OUT / "t6_stability.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n"
                "\\setlength{\\tabcolsep}{4.5pt}\n")
        f.write("\\caption{Stability of the direction under repeated splitting, and "
                  "the equivalence verdict against the pre-declared bound on a "
                  "practically meaningful effect.}\n")
        f.write("\\label{tab:stability}\n")
        f.write("\\begin{tabular}{lrrrlr}\n\\toprule\n")
        f.write("Model & $n$ & Paired diff. & Sign agreement & 95\\% CI & "
                  "$|\\Delta| \\ge 0.05$ excluded \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['n']} & {r['mean']:+.4f} & {r['agree']:.3f} & "
                    f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] & {r['rope']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the labels are English in both versions, so only the Japanese figure needs a
    # CJK face ahead of the lab standard for the model names it prints
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "font.size": 9})
    FIG = OUT
    FIG.mkdir(parents=True, exist_ok=True)
    # a model with ten seeds contributes a single point, which reads as a defect
    # rather than as information, so the curve is drawn for the confirmatory runs
    shown = [r for r in rows if r["n"] >= 30]
    fig, axes = plt.subplots(1, len(shown), figsize=(3.0 * len(shown), 2.7),
                             sharey=True)
    if len(shown) == 1:
        axes = [axes]
    for ax, r in zip(axes, shown):
        d = dict(MODELS)[r["model"]]
        dd = diffs(d, list(seeds_for(d)))
        rng = random.Random(7)
        xs, ms, los, his = [], [], [], []
        step = 5 if len(dd) <= 40 else 10
        for k in range(10, len(dd) + 1, step):
            sub = dd[:k]
            lo, hi = boot_ci(sub, rng, iters=1500)
            xs.append(k)
            ms.append(st.mean(sub))
            los.append(lo)
            his.append(hi)
        ax.plot(xs, ms, color="k", linewidth=1.2, marker="o", markersize=3)
        ax.fill_between(xs, los, his, color="0.7", alpha=0.45, linewidth=0)
        ax.axhline(0, color="0.3", linewidth=0.8)
        ax.axhline(ROPE, color="r", linewidth=0.7, linestyle="-.", alpha=0.6)
        ax.set_title(r["model"], fontsize=9)
        ax.set_xlabel("seeds used")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("asymmetry $\\hat\\Delta_n$")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cumulative.pdf", bbox_inches="tight")
    json.dump(rows, open(OUT / "stability.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t6_stability.tex and {FIG}/fig4_cumulative.pdf")


if __name__ == "__main__":
    main()
