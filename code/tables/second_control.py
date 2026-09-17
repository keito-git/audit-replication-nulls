"""The audit's control asks for a property that is hard to read; a second one is
trivial to read. If agents ignore both, they are rejecting meaningless
preferences rather than failing to compute a feature.

Declared in the framework lock §7 before the confirmatory seeds were taken.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, norm_reinf  # noqa: E402

MODELS = [("Mistral-Small (24B)", "gate_runs_mistralai-mistral-small-24b-instruct-2501"),
          ("Solar-Pro4 (22B)", "gate_runs"),
          ("Llama-3.1 (8B)", "gate_runs_meta-llama-llama-3-1-8b-instruct")]
BASE = 30 / 70


def stats(d, cond, seeds):
    """Amplification, adherence, and the share of aligned items the agents were
    actually offered.

    The slate share is reported because the two controls did not run in exactly
    the same environment: the tally gained a third feature on 2026-09-15 so that
    the initial-letter preference had a reinforcement channel of its own, which
    the h runs, made earlier, did not have. Comparing adherence against the
    slate each condition faced, rather than against the static pool ratio,
    removes that difference from the comparison."""
    amp, adh, slate = [], [], []
    for s in seeds:
        f = DATA / d / f"{cond}_s{s}.json"
        if not f.exists():
            continue
        L = json.load(open(f, encoding="utf-8"))
        amp.append(norm_reinf(L)[0])
        p = [x for rd in L["rounds"] for x in rd["pick_aligned"] if x is not None]
        if p:
            adh.append(st.mean(p))
        slate += [x for rd in L["rounds"] for x in rd["slate_aligned"]]
    return (st.mean(amp) if amp else float("nan"),
            st.mean(adh) if adh else float("nan"), len(amp),
            st.mean(slate) if slate else float("nan"))


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    rows = []
    print(f"{'model':22s} {'control':>12} {'n':>4} {'amp':>9} "
          f"{'adherence':>10} {'slate':>10}")
    for name, d in MODELS:
        for cond, lab in (("pref_h", "h frequency"),
                          ("pref_a", "Initial letter")):
            a, c, n, sl = stats(d, cond, range(1, 31))
            rows.append({"model": name, "control": lab, "n": n, "amp": a,
                         "adh": c, "slate": sl})
            print(f"{name:22s} {lab:>12} {n:4d} {a:+9.4f} {c:10.3f} {sl:10.3f}")

    with open(OUT / "t10_controls.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{The two meaningless controls side by side. The initial "
                  "share of aligned items is $30/70 = 0.429$ in both. The slate share "
                  "is the share of aligned items each condition was actually offered, "
                  "and it is what adherence should be read against.}\n")
        f.write("\\label{tab:controls}\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Model & Control & $n$ & Normalised amplification & Slate share & "
                  "Adherence \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['control']} & {r['n']} & "
                    f"{r['amp']:+.4f} & {r['slate']:.3f} & {r['adh']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump(rows, open(OUT / "controls.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t10_controls.tex")


if __name__ == "__main__":
    main()
