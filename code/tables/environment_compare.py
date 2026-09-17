"""Does the verdict survive a change of recommender?

Declared in the framework lock §7 before the confirmatory seeds were taken.

The first environment samples in proportion to a tally. A reviewer can fairly
call that a toy, and the objection matters: if the audit-replication verdict is
a property of the agents it should survive a different recommender, and if it
does not, then validating an LLM social simulation depends on the environment
model and not only on the agent model. Either answer is worth reporting.

The bandit shares the frozen pool, the slate size, the exploration budget, the
prompts, the seeds and the criteria. Only the ranking rule differs.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, seeds_for, norm_reinf, exact_one_sided  # noqa: E402

MODEL = "Mistral-Small (24B)"
TALLY_DIR = "gate_runs_mistralai-mistral-small-24b-instruct-2501"
BANDIT_DIR = "bandit_runs_mistralai-mistral-small-24b-instruct-2501"
CONDS = ["pref_neg", "pref_pos", "pref_h"]


def seeds_in(d):
    n = 0
    while all((DATA / d / f"{c}_s{n + 1}.json").exists() for c in CONDS) and n < 100:
        n += 1
    return list(range(1, n + 1))


def summarise(d, seeds):
    out = {}
    for c in CONDS:
        vals = [norm_reinf(json.load(open(DATA / d / f"{c}_s{s}.json",
                                          encoding="utf-8")))[0] for s in seeds]
        out[c] = vals
    dd, p = exact_one_sided(out["pref_neg"], out["pref_pos"])
    return {"neg": st.mean(out["pref_neg"]), "pos": st.mean(out["pref_pos"]),
            "h": st.mean(out["pref_h"]), "diff": dd, "p": p, "n": len(seeds)}


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)

    b_seeds = seeds_in(BANDIT_DIR)
    if not b_seeds:
        print("no bandit runs found")
        return
    # compare like with like: the tally environment restricted to the same seeds
    t_stats = summarise(TALLY_DIR, b_seeds)
    b_stats = summarise(BANDIT_DIR, b_seeds)
    t_full = summarise(TALLY_DIR, seeds_in(TALLY_DIR))

    rows = [("Tally (matched seeds)", t_stats),
            ("Contextual bandit", b_stats),
            ("Tally (all seeds)", t_full)]
    print(f"{'environment':28s} {'n':>4} {'neg':>8} {'pos':>8} {'h':>8} "
          f"{'diff':>9} {'p':>8}")
    for lab, s in rows:
        print(f"{lab:28s} {s['n']:4d} {s['neg']:+8.3f} {s['pos']:+8.3f} "
              f"{s['h']:+8.3f} {s['diff']:+9.4f} {s['p']:8.4f}")

    with open(OUT / "t9_environment.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Verdict when the recommender is replaced. The pool, the "
                  "slate size, the exploration rate, the prompts, the seed sequence and "
                  "the pass criteria are shared.}\n")
        f.write("\\label{tab:environment}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Recommender & $n$ & Neg. & Pos. & Meaningless & Diff. & "
                  "$p_{A2}$ \\\\\n\\midrule\n")
        for lab, s in rows:
            f.write(f"{lab} & {s['n']} & {s['neg']:+.3f} & {s['pos']:+.3f} & "
                    f"{s['h']:+.3f} & {s['diff']:+.4f} & {s['p']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump({lab: s for lab, s in rows},
              open(OUT / "environment.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t9_environment.tex")


if __name__ == "__main__":
    main()
