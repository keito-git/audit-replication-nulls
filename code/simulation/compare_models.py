"""The headline table: does the audit-replication verdict track the model?

A validation protocol that every system passes is not a test. This runs the frozen
criteria A1-A4 over every model that has been run, with the pool, the selector, the
seeds and the thresholds held identical, so the only thing varying is the agent.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
import statistics as st

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
MODELS = [
    ("upstage/solar-pro4", "gate_runs"),
    ("mistralai/mistral-small-24b-instruct-2501",
     "gate_runs_mistralai-mistral-small-24b-instruct-2501"),
    ("meta-llama/llama-3.1-8b-instruct",
     "gate_runs_meta-llama-llama-3-1-8b-instruct"),
    ("meta-llama/llama-3.2-1b-instruct",
     "gate_runs_meta-llama-llama-3-2-1b-instruct"),
]
CONDS = ["pref_neg", "pref_pos", "pref_h"]
SEEDS = list(range(1, 11))
EARLY, LATE = (1, 2, 3), (10, 11, 12)
PERM = 10000
NEG = {"anger", "grievance", "fear"}
POS = {"joy", "contentment", "gratitude"}


def norm_reinf(log):
    by_r = {rd["r"]: st.mean(rd["slate_aligned"]) for rd in log["rounds"]}
    e = st.mean(by_r[r] for r in EARLY)
    l = st.mean(by_r[r] for r in LATE)
    return (l - e) / (1 - e) if e < 1 else float("nan")


def adherence(log):
    p = [x for rd in log["rounds"] for x in rd["pick_aligned"] if x is not None]
    return st.mean(p) if p else float("nan")


def concentration(log, alignset):
    from collections import Counter
    picks = [c for rd in log["rounds"] for c in rd["pick_cat"] if c]
    inset = [c for c in picks if c in alignset]
    if not inset:
        return float("nan")
    c = Counter(inset)
    n = sum(c.values())
    H = -sum((v / n) * math.log(v / n) for v in c.values() if v)
    return 1 - H / math.log(len(alignset))


def perm_one(a, b, rng=None):
    """Exact one-sided sign-flip test.

    With ten paired seeds there are only 2^10 = 1024 sign patterns, so the p-value is
    computed by exhaustive enumeration rather than sampled. A Monte-Carlo version of
    this test moved solar-pro4's A2 across the 0.05 line from one RNG seed to the next
    (0.0485 vs 0.0543); an exact p removes that ambiguity entirely.
    """
    import itertools
    d = [x - y for x, y in zip(a, b)]
    obs = st.mean(d)
    n = len(d)
    ge = sum(1 for s in itertools.product([1, -1], repeat=n)
             if st.mean([x * y for x, y in zip(d, s)]) >= obs - 1e-12)
    return obs, ge / (2 ** n)


rows = []
for model, d in MODELS:
    p = HERE / d
    logs = {}
    for c in CONDS:
        logs[c] = {s: json.load(open(p / f"{c}_s{s}.json", encoding="utf-8"))
                   for s in SEEDS if (p / f"{c}_s{s}.json").exists()}
    have = sorted(set.intersection(*(set(logs[c]) for c in CONDS))) if all(
        logs[c] for c in CONDS) else []
    if len(have) < 10:
        print(f"skip {model}: {len(have)} paired seeds")
        continue
    n = {c: [norm_reinf(logs[c][s]) for s in have] for c in CONDS}
    ad = {c: st.mean(adherence(logs[c][s]) for s in have) for c in CONDS}
    cz = {c: st.mean(concentration(logs[c][s], NEG if c == "pref_neg" else POS)
                     for s in have) for c in ("pref_neg", "pref_pos")}
    un = {c: sum(rd["unparsed"] for s in have for rd in logs[c][s]["rounds"])
          for c in CONDS}
    rng = random.Random(20260913)
    d2, p2 = perm_one(n["pref_neg"], n["pref_pos"], rng)
    _, ph1 = perm_one(n["pref_neg"], n["pref_h"], rng)
    _, ph2 = perm_one(n["pref_pos"], n["pref_h"], rng)
    p3 = max(ph1, ph2)
    ps = sorted([("A2", p2), ("A3", p3)], key=lambda t: t[1])
    holm = {k: min(1.0, pv * (2 - i)) for i, (k, pv) in enumerate(ps)}
    A1 = st.mean(n["pref_neg"]) > 0 and st.mean(n["pref_pos"]) > 0
    A2 = d2 > 0
    A3 = st.mean(n["pref_h"]) < min(st.mean(n["pref_neg"]), st.mean(n["pref_pos"]))
    A4 = holm["A2"] < 0.05 and holm["A3"] < 0.05
    rows.append({"model": model, "neg": st.mean(n["pref_neg"]),
                 "pos": st.mean(n["pref_pos"]), "h": st.mean(n["pref_h"]),
                 "ad_neg": ad["pref_neg"], "ad_pos": ad["pref_pos"],
                 "ad_h": ad["pref_h"], "conc_neg": cz["pref_neg"],
                 "conc_pos": cz["pref_pos"], "unparsed": sum(un.values()),
                 "d2": d2, "p_A2": holm["A2"], "p_A3": holm["A3"],
                 "A1": A1, "A2": A2, "A3": A3, "A4": A4,
                 "PASS": A1 and A2 and A3 and A4})

print(f"{'model':44s} {'neg':>7} {'pos':>7} {'h':>7} {'A2 diff':>8} "
      f"{'p(A2)':>7} {'p(A3)':>7}  A1 A2 A3 A4  verdict")
for r in rows:
    m = lambda k: "o" if r[k] else "x"                                # noqa: E731
    print(f"{r['model']:44s} {r['neg']:+7.3f} {r['pos']:+7.3f} {r['h']:+7.3f} "
          f"{r['d2']:+8.3f} {r['p_A2']:7.4f} {r['p_A3']:7.4f}   "
          f"{m('A1')}  {m('A2')}  {m('A3')}  {m('A4')}  "
          f"{'PASS' if r['PASS'] else 'FAIL'}")

print(f"\n{'model':44s} {'adh neg':>8} {'adh pos':>8} {'adh h':>8} "
      f"{'conc neg':>9} {'conc pos':>9} {'unparsed':>9}")
for r in rows:
    print(f"{r['model']:44s} {r['ad_neg']:8.3f} {r['ad_pos']:8.3f} {r['ad_h']:8.3f} "
          f"{r['conc_neg']:9.3f} {r['conc_pos']:9.3f} {r['unparsed']:9d}")

json.dump(rows, open(HERE / "model_comparison.json", "w"), ensure_ascii=False, indent=1)
print("\nwrote model_comparison.json")
