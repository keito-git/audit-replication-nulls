"""Descriptive statistics of the frozen content pool.

The pool is the only material the simulation reads, so a reader who wants to
judge whether the three assigned preferences are comparable needs its
composition in one place: how many items carry each category, how the arbitrary
textual feature is distributed over those categories, and how long the items
are. The derived indices are recomputed here exactly as gate.load_pool
recomputes them, from the frozen file, so the table cannot drift from the runs.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st

POOL = pathlib.Path(__file__).resolve().parents[2] / "data" / "pool.json"
CATS = ["anger", "grievance", "fear", "joy", "contentment", "gratitude", "neutral"]
NEG = ["anger", "grievance", "fear"]
POS = ["joy", "contentment", "gratitude"]
PER_CAT = 10
LABEL = {c: c.capitalize() for c in CATS}


def load_pool():
    p = json.load(open(POOL, encoding="utf-8"))
    order = sorted(p["items"], key=lambda it: -it["h"])
    for k, it in enumerate(order):
        it["h_sept"] = k // PER_CAT
    return p


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    items = pool["items"]
    rows = []
    for c in CATS:
        sub = [it for it in items if it["category"] == c]
        rows.append({
            "cat": c,
            "n": len(sub),
            "h": st.mean(it["h"] for it in sub),
            "top3": sum(1 for it in sub if it["h_sept"] < 3),
            "words": st.mean(len(it["text"].split()) for it in sub),
        })
    total = {"n": len(items), "h": st.mean(it["h"] for it in items),
             "top3": sum(1 for it in items if it["h_sept"] < 3),
             "words": st.mean(len(it["text"].split()) for it in items)}

    print(f"{'category':12s} {'n':>3} {'mean h':>8} {'top3 sept':>10} {'words':>7}")
    for r in rows:
        print(f"{r['cat']:12s} {r['n']:3d} {r['h']:8.4f} {r['top3']:10d} "
              f"{r['words']:7.1f}")
    print(f"{'total':12s} {total['n']:3d} {total['h']:8.4f} {total['top3']:10d} "
          f"{total['words']:7.1f}")
    print(f"h cut for the meaningless preference: {pool['h_cut']:.4f}")

    with open(OUT / "t14_pool.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Composition of the frozen content pool.}\n")
        f.write("\\label{tab:pool}\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Preference & Category & Items & Mean $h$ & Items in top three "
                "septiles & Mean words \\\\\n\\midrule\n")
        for r in rows:
            if r["cat"] in NEG:
                pref = "Negative"
            elif r["cat"] in POS:
                pref = "Positive"
            else:
                pref = "Unaligned"
            f.write(f"{pref} & {LABEL[r['cat']]} & {r['n']} & {r['h']:.4f} & "
                    f"{r['top3']} & {r['words']:.1f} \\\\\n")
        f.write("\\midrule\n")
        f.write("Total & --- & "
                + f"{total['n']} & {total['h']:.4f} & {total['top3']} & "
                + f"{total['words']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    json.dump({"rows": rows, "total": total, "h_cut": pool["h_cut"]},
              open(OUT / "pool_stats.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t14_pool.tex")


if __name__ == "__main__":
    main()
