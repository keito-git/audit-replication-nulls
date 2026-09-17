"""Does the reproduction survive a change in how the preference is conveyed?

Declared in the framework lock §7 before the confirmatory seeds were taken.

The audit's sock puppets were programmed with a preference. A real user is not
told what to prefer; the preference shows up in what they have already opened.
Three forms are compared: stated outright, written as a disposition, and never
stated at all but carried by the agent's own history of choices.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, norm_reinf, exact_one_sided  # noqa: E402

BASE = "gate_runs_mistralai-mistral-small-24b-instruct-2501"
STYLES = [(BASE, "Explicit"),
          (BASE + "_persona", "Persona"),
          (BASE + "_behaviour", "Behaviour history")]
CONDS = ["pref_neg", "pref_pos", "pref_h"]
SEEDS = range(1, 31)
BASE_RATE = 30 / 70


def block(d):
    out = {}
    for c in CONDS:
        amp, adh = [], []
        for s in SEEDS:
            L = json.load(open(DATA / d / f"{c}_s{s}.json", encoding="utf-8"))
            amp.append(norm_reinf(L)[0])
            p = [x for rd in L["rounds"] for x in rd["pick_aligned"] if x is not None]
            if p:
                adh.append(st.mean(p))
        out[c] = (st.mean(amp), st.mean(adh))
    neg = [norm_reinf(json.load(open(DATA / d / f"pref_neg_s{s}.json",
                                     encoding="utf-8")))[0] for s in SEEDS]
    pos = [norm_reinf(json.load(open(DATA / d / f"pref_pos_s{s}.json",
                                     encoding="utf-8")))[0] for s in SEEDS]
    dd, p = exact_one_sided(neg, pos)
    return out, dd, p


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    rows = []
    print(f"{'style':>10} {'neg':>9} {'pos':>9} {'meaningless':>12} "
          f"{'diff':>9} {'p':>8} {'adherence':>10}")
    for d, lab in STYLES:
        o, dd, p = block(d)
        rows.append({"style": lab, "neg": o["pref_neg"][0], "pos": o["pref_pos"][0],
                     "h": o["pref_h"][0], "diff": dd, "p": p,
                     "adh_neg": o["pref_neg"][1], "adh_h": o["pref_h"][1]})
        print(f"{lab:>10} {o['pref_neg'][0]:+9.4f} {o['pref_pos'][0]:+9.4f} "
              f"{o['pref_h'][0]:+12.4f} {dd:+9.4f} {p:8.4f} {o['pref_neg'][1]:10.3f}")

    with open(OUT / "t11_prompts.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Verdict when the preference is conveyed differently. The "
                  "initial share of aligned items is $0.429$.}\n")
        f.write("\\label{tab:prompts}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Conveyance & Neg. & Pos. & Meaningless & Diff. & $p_{A2}$ & "
                  "Adherence \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['style']} & {r['neg']:+.4f} & {r['pos']:+.4f} & "
                    f"{r['h']:+.4f} & {r['diff']:+.4f} & {r['p']:.4f} & "
                    f"{r['adh_neg']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump(rows, open(OUT / "prompts.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t11_prompts.tex")


if __name__ == "__main__":
    main()
