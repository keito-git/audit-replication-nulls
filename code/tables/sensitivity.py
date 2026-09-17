"""Does the verdict depend on choices the framework had to make arbitrarily?

Declared in the framework lock §6 before the confirmatory seeds were taken.

Two of those choices are visible in the stored runs and need no further calls:
the window used to measure amplification, and the size of the aligned set that
the normalisation divides by. A third, the exploration rate, changes the runs
themselves and is therefore left to a separate sweep.

The point is not to find a setting that helps. It is to show that the verdict
is the same across the settings a reasonable person would have picked.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, MODELS, seeds_for, exact_one_sided  # noqa: E402

WINDOWS = [((1, 2, 3), (10, 11, 12)), ((1, 2, 3, 4), (9, 10, 11, 12)),
           ((1, 2), (11, 12))]


def amp(log, early, late):
    by = {rd["r"]: st.mean(rd["slate_aligned"]) for rd in log["rounds"]}
    e = st.mean(by[r] for r in early)
    l = st.mean(by[r] for r in late)
    return (l - e) / (1 - e) if e < 1 else float("nan")


def slope(log):
    """Amplification as the slope of the aligned share over rounds, which does
    not depend on a window at all."""
    by = {rd["r"]: st.mean(rd["slate_aligned"]) for rd in log["rounds"]}
    rs = sorted(by)
    xb = st.mean(rs)
    yb = st.mean(by[r] for r in rs)
    num = sum((r - xb) * (by[r] - yb) for r in rs)
    den = sum((r - xb) ** 2 for r in rs)
    return num / den if den else float("nan")


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = ["1--3 / 10--12", "1--4 / 9--12", "1--2 / 11--12", "Slope"]
    print(f"{'model':22s} {'n':>4} " + "".join(f"{l:>16}" for l in labels))
    for name, d in MODELS:
        seeds = list(seeds_for(d))
        if len(seeds) < 10:
            continue
        logs = {c: {s: json.load(open(DATA / d / f"{c}_s{s}.json", encoding="utf-8"))
                    for s in seeds} for c in ("pref_neg", "pref_pos")}
        cells = []
        for early, late in WINDOWS:
            neg = [amp(logs["pref_neg"][s], early, late) for s in seeds]
            pos = [amp(logs["pref_pos"][s], early, late) for s in seeds]
            dd, p = exact_one_sided(neg, pos)
            cells.append((dd, p))
        neg = [slope(logs["pref_neg"][s]) for s in seeds]
        pos = [slope(logs["pref_pos"][s]) for s in seeds]
        dd, p = exact_one_sided(neg, pos)
        cells.append((dd, p))
        rows.append({"model": name, "n": len(seeds), "cells": cells})
        print(f"{name:22s} {len(seeds):4d} " +
              "".join(f"  {c[0]:+.4f} p{c[1]:.3f}" for c in cells))

    with open(OUT / "t8_sensitivity.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Paired difference for A2 and its one-sided $p$ under "
                  "different aggregation windows. The rightmost column measures the "
                  "same quantity as a slope, which uses no window at all.}\n")
        f.write("\\label{tab:sensitivity}\n")
        f.write("\\begin{tabular}{lr" + "r" * len(labels) + "}\n\\toprule\n")
        f.write("Model" + " & $n$ & " + " & ".join(labels)
                + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['n']} & "
                    + " & ".join(f"{c[0]:+.4f}" for c in r["cells"]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump(rows, open(OUT / "sensitivity.json",
                         "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t8_sensitivity.tex")
    print("\nthe verdict should not depend on the window; a model whose sign "
          "changes with the window has a window-dependent verdict")


if __name__ == "__main__":
    main()
