"""Null-agent test of the platform: replace the LLM with a chooser that follows the
assigned preference with a fixed, identical probability in every arm.

If the platform's feature space represents the three preferences symmetrically,
the three arms must show the SAME reinforcement at the same adherence. Any gap is
a property of the representation, not of the agents. This costs no LLM calls and
should have been run before spending on the gate.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
import gate  # noqa: E402  -- reuse the exact selector under test

SEEDS = list(range(1, 21))
EARLY, LATE = (1, 2, 3), (10, 11, 12)


def run(cond, seed, pool, adherence, feature_fn):
    items = pool["items"]
    tally = [{b: 0.0 for b in feature_fn.buckets_all} for _ in range(gate.N_AGENTS)]
    per_r = {}
    for r in range(1, gate.N_ROUNDS + 1):
        al = []
        for i in range(gate.N_AGENTS):
            rng = random.Random(seed * 100003 + r * 997 + i)
            n_rand = sum(1 for _ in range(gate.SLATE) if rng.random() < gate.EPS)
            chosen, used = [], set()
            rest = items[:]
            rng.shuffle(rest)
            for it in rest[:n_rand]:
                chosen.append(it)
                used.add(it["id"])
            cand = [it for it in items if it["id"] not in used]
            while len(chosen) < gate.SLATE and cand:
                w = [1.0 + sum(tally[i][b] for b in feature_fn(it)) for it in cand]
                tot = sum(w)
                x = rng.random() * tot
                acc = 0.0
                for it, wi in zip(cand, w):
                    acc += wi
                    if acc >= x:
                        chosen.append(it)
                        cand.remove(it)
                        break
            al.append(sum(1 for it in chosen if gate.aligned(cond, it)) / len(chosen))
            # the null agent: follows the preference with probability `adherence`
            good = [it for it in chosen if gate.aligned(cond, it)]
            bad = [it for it in chosen if not gate.aligned(cond, it)]
            pick = (rng.choice(good) if good and rng.random() < adherence
                    else (rng.choice(bad) if bad else rng.choice(chosen)))
            for b in feature_fn(pick):
                tally[i][b] += 1.0
        per_r[r] = st.mean(al)
    e = st.mean(per_r[r] for r in EARLY)
    l = st.mean(per_r[r] for r in LATE)
    return (l - e) / (1 - e)


class Current:
    """The feature space actually used in the gate: 7 categories + one high_h flag."""
    buckets_all = gate.CATS + ["high_h"]

    def __call__(self, it):
        return gate.buckets(it)


class Septile:
    """Repair: the arbitrary textual feature is discretised at the SAME granularity
    as the emotional feature -- 7 h-septiles of 10 items mirroring 7 categories of
    10 items -- so each preference is carried by exactly three 10-item features."""
    buckets_all = gate.CATS + [f"h{k}" for k in range(7)]

    def __init__(self, items):
        order = sorted(items, key=lambda it: -it["h"])
        self.sept = {it["id"]: k // 10 for k, it in enumerate(order)}

    def __call__(self, it):
        return [it["category"], f"h{self.sept[it['id']]}"]


def main():
    pool = json.load(open(HERE / "pool.json", encoding="utf-8"))
    items = pool["items"]
    sept = Septile(items)
    # keep the aligned set for pref_h identical (top 30 by h) under both spaces
    for space_name, fn in (("current (7 cats + 1 high_h flag)", Current()),
                           ("repaired (7 cats + 7 h-septiles)", sept)):
        print(f"\n=== platform feature space: {space_name} ===")
        print(f"{'adherence':>10} " + "".join(f"{c:>12}" for c in gate.CONDS_ORDER))
        for adh in (0.55, 0.75, 1.00):
            row = []
            for cond in gate.CONDS_ORDER:
                vals = [run(cond, s, pool, adh, fn) for s in SEEDS]
                row.append(st.mean(vals))
            spread = max(row) - min(row)
            print(f"{adh:10.2f} " + "".join(f"{x:+12.3f}" for x in row) +
                  f"   spread {spread:+.3f}")


if __name__ == "__main__":
    main()
