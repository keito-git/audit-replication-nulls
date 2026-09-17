"""How much of the amplification is explained by compliance and concentration.

Defined in the framework lock before the confirmatory seeds were taken.

The diagnostics table already shows that compliance separates the models and
that the negative condition concentrates on one category while the positive one
spreads. This turns those two observations into a decomposition: per seed,
regress the amplification on the agent-side quantities and report what is left.
Whatever the agent side does not explain is the most the environment can be
credited with, which is the same logic as the attribution ladder measured a
different way.

No LLM calls; everything is read off the stored runs.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys
from collections import Counter

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, MODELS, CONDS, seeds_for, norm_reinf, NEG, POS  # noqa: E402


def compliance(log):
    p = [x for rd in log["rounds"] for x in rd["pick_aligned"] if x is not None]
    return st.mean(p) if p else float("nan")


def concentration(log, alignset):
    inset = [c for rd in log["rounds"] for c in rd["pick_cat"] if c and c in alignset]
    if not inset:
        return float("nan")
    c = Counter(inset)
    n = sum(c.values())
    H = -sum((v / n) * math.log(v / n) for v in c.values() if v)
    return 1 - H / math.log(len(alignset))


def ols(y, X):
    """Least squares with an intercept, solved by Gauss-Jordan on the normal
    equations. The design is tiny, so this keeps the file dependency-free."""
    n, k = len(y), len(X[0]) + 1
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    rows = [[1.0] + list(x) for x in X]
    for r, yi in zip(rows, y):
        for i in range(k):
            b[i] += r[i] * yi
            for j in range(k):
                A[i][j] += r[i] * r[j]
    M = [A[i][:] + [b[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(k):
            if r != c and abs(M[r][c]) > 1e-15:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    beta = [M[i][k] for i in range(k)]
    fit = [sum(bi * ri for bi, ri in zip(beta, r)) for r in rows]
    ybar = st.mean(y)
    ss_res = sum((a - f) ** 2 for a, f in zip(y, fit))
    ss_tot = sum((a - ybar) ** 2 for a in y)
    return beta, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'model':22s} {'n':>4} {'R2 compl':>9} {'R2 +conc':>9} "
          f"{'b_compl':>9} {'b_conc':>9}")
    for name, d in MODELS:
        seeds = list(seeds_for(d))
        if len(seeds) < 10:
            continue
        y, X1, X2 = [], [], []
        for s in seeds:
            for cond, aligned in (("pref_neg", NEG), ("pref_pos", POS)):
                log = json.load(open(DATA / d / f"{cond}_s{s}.json", encoding="utf-8"))
                r = norm_reinf(log)[0]
                c = compliance(log)
                h = concentration(log, aligned)
                if r == r and c == c and h == h:
                    y.append(r)
                    X1.append([c])
                    X2.append([c, h])
        m1 = ols(y, X1)
        m2 = ols(y, X2)
        if not m1 or not m2:
            continue
        rows.append({"model": name, "n": len(seeds), "n_obs": len(y),
                     "r2_compliance": m1[1], "r2_both": m2[1],
                     "b_compliance": m2[0][1], "b_concentration": m2[0][2]})
        print(f"{name:22s} {len(seeds):4d} {m1[1]:9.3f} {m2[1]:9.3f} "
              f"{m2[0][1]:+9.3f} {m2[0][2]:+9.3f}")

    with open(OUT / "t7_mediation.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n"
                "\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Normalised amplification regressed on adherence and "
                  "concentration. The coefficient of determination is the share "
                  "explained by the agent-side quantities alone.}\n")
        f.write("\\label{tab:mediation}\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Model & $n$ & Observations & $R^2$ adherence & "
                  "$R^2$ adherence$+$concentration & "
                  "$\\beta_{\\text{concentration}}$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['n']} & {r['n_obs']} & "
                    f"{r['r2_compliance']:.3f} & {r['r2_both']:.3f} & "
                    f"{r['b_concentration']:+.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump(rows, open(OUT / "mediation.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t7_mediation.tex")


if __name__ == "__main__":
    main()
