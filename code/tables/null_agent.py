"""Table for the null-agent test: does the feature space favour a condition?

The generative model is replaced by a chooser that follows the assigned
preference with a fixed probability, identical in every condition. If the
representation is symmetric the conditions must agree, so any spread is a
property of the representation.

Two representations are compared. Under the single-bucket one the thirty items
aligned with the meaningless preference share one feature; under the matched one
that feature is cut into septiles, so each of the three preferences is carried
by three features of ten items.

No LLM calls.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import statistics as st
import sys



DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
SIM = pathlib.Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIM))
import gate  # noqa: E402

SEEDS = list(range(1, 21))
EARLY, LATE = (1, 2, 3), (10, 11, 12)
CONDS = ["pref_neg", "pref_pos", "pref_h"]
ADHERENCE = [0.55, 0.75, 1.00]


def single_bucket(it):
    """The thirty items aligned with the meaningless preference share one feature,
    and the remaining forty carry no feature of that kind at all."""
    return [it["category"]] + (["high_h"] if it["high_h"] else [])


def matched(it):
    """The same feature at the granularity of the emotional one."""
    return [it["category"], f"h{it['h_sept']}"]


SPACES = [("single", "Single bucket", single_bucket,
           gate.CATS + ["high_h"]),
          ("matched", "Matched granularity", matched,
           gate.CATS + [f"h{k}" for k in range(7)])]


def run(cond, seed, pool, adherence, feature_fn, buckets_all):
    items = pool["items"]
    tally = [{b: 0.0 for b in buckets_all} for _ in range(gate.N_AGENTS)]
    per_r = {}
    for r in range(1, gate.N_ROUNDS + 1):
        al = []
        for i in range(gate.N_AGENTS):
            rng = random.Random(seed * 100003 + r * 997 + i)
            n_rand = sum(1 for _ in range(gate.SLATE) if rng.random() < gate.EPS)
            chosen, used, rest = [], set(), items[:]
            rng.shuffle(rest)
            for it in rest[:n_rand]:
                chosen.append(it)
                used.add(it["id"])
            cand = [it for it in items if it["id"] not in used]
            while len(chosen) < gate.SLATE and cand:
                w = [1.0 + sum(tally[i][b] for b in feature_fn(it)) for it in cand]
                x = rng.random() * sum(w)
                acc = 0.0
                for it, wi in zip(cand, w):
                    acc += wi
                    if acc >= x:
                        chosen.append(it)
                        cand.remove(it)
                        break
            al.append(sum(1 for it in chosen if gate.aligned(cond, it)) / len(chosen))
            good = [it for it in chosen if gate.aligned(cond, it)]
            bad = [it for it in chosen if not gate.aligned(cond, it)]
            follow = rng.random() < adherence
            pool_pick = good if (follow and good) else (bad or chosen)
            pick = rng.choice(pool_pick)
            for b in feature_fn(pick):
                tally[i][b] += 1.0
        per_r[r] = st.mean(al)
    e = st.mean(per_r[r] for r in EARLY)
    l = st.mean(per_r[r] for r in LATE)
    return (l - e) / (1 - e) if e < 1 else float("nan")


def main():
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    OUT.mkdir(parents=True, exist_ok=True)
    pool = gate.load_pool()   # adds the septile and band indices
    rows = []
    print(f"{'space':22s} {'adh':>5} " + "".join(f"{c:>12}" for c in CONDS) + "   spread")
    for key, label, fn, buckets_all in SPACES:
        for adh in ADHERENCE:
            vals = [st.mean(run(c, s, pool, adh, fn, buckets_all) for s in SEEDS)
                    for c in CONDS]
            spread = max(vals) - min(vals)
            rows.append({"space": key, "label": label, "adherence": adh,
                         "vals": vals, "spread": spread})
            print(f"{label:22s} {adh:5.2f} " + "".join(f"{v:+12.3f}" for v in vals)
                  + f"   {spread:+.3f}")

    with open(OUT / "t12_nullagent.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{The null-agent test. The generative model is replaced by a "
                  "random chooser that follows the assigned preference with the same "
                  "probability in every condition, and the environment alone is run. "
                  "If the representation is symmetric the three conditions must agree, "
                  "so a spread is a property of the representation.}\n")
        f.write("\\label{tab:nullagent}\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Representation & Adherence & Neg. & Pos. & Meaningless & Spread "
                  "\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['label']} & {r['adherence']:.2f} & "
                    + " & ".join(f"{v:+.3f}" for v in r["vals"])
                    + f" & {r['spread']:+.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    json.dump(rows, open(OUT / "nullagent.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}/t12_nullagent.tex")


if __name__ == "__main__":
    main()
