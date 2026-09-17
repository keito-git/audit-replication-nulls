"""Table for the application arm: the regime for which no audit finding exists.

Every number is read off the stored runs through the same functions that
produced the verdict, so the table cannot drift from it.

No LLM calls.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics as st
import sys



DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
SIM = pathlib.Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM))
os.chdir(SIM)
import verdict2  # noqa: E402

ROWS = [("none", "Random delivery"),
        ("relevance", "Topic matching"),
        ("engagement", "Learned from selections"),
        ("affect", "Affect optimising"),
        ("volume_null", "Volume-matched random")]


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)
    logs = {c: verdict2.load(c) for c, _ in ROWS}
    verdict = json.load(open(DATA / "verdict2.json", encoding="utf-8"))

    rows = []
    for cond, label in ROWS:
        ls = logs[cond]
        seeds = sorted(ls)
        fm = st.mean(verdict2.feed_mean(ls[s]) for s in seeds)
        vm = st.mean(verdict2.vol_mean(ls[s]) for s in seeds)
        m1 = [verdict2.M1(ls[s]) for s in seeds]
        m1 = [x for x in m1 if x == x]
        rows.append({"cond": cond, "label": label, "n": len(seeds),
                     "feed": fm, "vol": vm,
                     "rho": st.mean(m1) if m1 else float("nan")})
        print(f"{label:24s} n={len(seeds)} feed {fm:+.3f} vol {vm:5.2f} "
              f"rho {rows[-1]['rho']:+.3f}")

    k1 = verdict["criteria"]["K1"]
    with open(OUT / "t13_application.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Application to a regime for which no audit finding "
                  "exists. The mean feed valence shows whether the treatment was "
                  "delivered, and the rank correlation is the statistic of the "
                  "prediction registered in advance.}\n")
        f.write("\\label{tab:application}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Condition & $n$ & Volume & Mean feed valence & "
                  "Rank correlation \\\\\n\\midrule\n")
        for r in rows:
            rho = "--" if r["rho"] != r["rho"] else f"{r['rho']:+.3f}"
            f.write(f"{r['label']} & {r['n']} & {r['vol']:.2f} & "
                    f"{r['feed']:+.3f} & {rho} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump({"rows": rows, "K1": k1},
              open(OUT / "application.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t13_application.tex")


if __name__ == "__main__":
    main()
