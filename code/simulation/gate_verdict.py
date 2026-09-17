"""Applies the pre-registered rule mechanically.

Directions are fixed in advance by Habib & Nithyanand (arXiv:2501.15048), so the
permutation tests are one-sided. The verdict is written before any interpretation.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import re
import random
import statistics as st

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
CONDS = ["pref_neg", "pref_pos", "pref_h"]
SEEDS = list(range(1, int(os.environ.get("AUDITREP_SEEDS", "10")) + 1))
EARLY, LATE = (1, 2, 3), (10, 11, 12)
PERM = 10000


def rundir():
    m = os.environ.get("AUDITREP_MODEL", "upstage/solar-pro4")
    if m == "upstage/solar-pro4":
        return HERE / "gate_runs"
    return HERE / ("gate_runs_" + re.sub(r"[^a-z0-9]+", "-", m.lower()).strip("-"))


def load(cond):
    out = {}
    for s in SEEDS:
        f = rundir() / f"{cond}_s{s}.json"
        if f.exists():
            out[s] = json.load(open(f, encoding="utf-8"))
    return out


def shares(log):
    by_r = {rd["r"]: st.mean(rd["slate_aligned"]) for rd in log["rounds"]}
    early = st.mean(by_r[r] for r in EARLY)
    late = st.mean(by_r[r] for r in LATE)
    raw = late - early
    norm = raw / (1 - early) if early < 1 else float("nan")
    picks = [p for rd in log["rounds"] for p in rd["pick_aligned"] if p is not None]
    return {"early": early, "late": late, "raw": raw, "norm": norm,
            "adherence": st.mean(picks) if picks else float("nan"),
            "by_round": by_r,
            "unparsed": sum(rd["unparsed"] for rd in log["rounds"])}


def perm_one_sided(a, b, rng=None):
    """H1: mean(a) > mean(b), paired by seed. Exact one-sided sign-flip test.

    Changed from Monte Carlo to exhaustive enumeration on 2026-09-13. With ten
    paired seeds there are only 2^10 = 1024 sign patterns, so the p-value is exact.
    The sampled version moved solar-pro4's A2 across the 0.05 line between RNG
    seeds (0.0485 and 0.0543 for the same data); the exact value is 0.0518.
    """
    import bisect

    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    obs = st.mean(d)
    target = obs * n - 1e-9

    def sums(xs):
        out = [0.0]
        for v in xs:
            out = [s + v for s in out] + [s - v for s in out]
        return out

    left = sums(d[: n // 2])
    right = sorted(sums(d[n // 2:]))
    ge = sum(len(right) - bisect.bisect_left(right, target - s) for s in left)
    return obs, ge / (2 ** n)



# ---------------------------------------------------------------- diagnostics
# Not part of A1-A4. Reported to say how much of any asymmetry is already implied
# by which categories the agents picked, as opposed to anything further.

def concentration(log, alignset):
    import math
    from collections import Counter
    picks = [c for rd in log["rounds"] for c in rd["pick_cat"] if c]
    inset = [c for c in picks if c in alignset]
    if not inset:
        return float("nan"), {}
    c = Counter(inset)
    n = sum(c.values())
    H = -sum((v / n) * math.log(v / n) for v in c.values() if v)
    return 1 - H / math.log(len(alignset)), {k: v / n for k, v in c.items()}


def matched_null(cond, seed, pool, adherence, catdist):
    """A random chooser carrying the observed adherence and category distribution of
    this arm. If the observed reinforcement does not exceed this, the asymmetry is
    fully accounted for by which categories were picked."""
    import gate
    items = pool["items"]
    tally = [{b: 0.0 for b in gate.CATS + [f"h{k}" for k in range(len(gate.CATS))]}
             for _ in range(gate.N_AGENTS)]
    per_r = {}
    cats = list(catdist) or None
    wts = [catdist[c] for c in cats] if cats else None
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
                w = [1.0 + sum(tally[i][b] for b in gate.buckets(it)) for it in cand]
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
            good = [it for it in chosen if gate.aligned(cond, it)]
            bad = [it for it in chosen if not gate.aligned(cond, it)]
            pick = None
            if good and rng.random() < adherence:
                if cats:
                    want = rng.choices(cats, weights=wts)[0]
                    same = [it for it in good if it["category"] == want]
                    pick = rng.choice(same) if same else rng.choice(good)
                else:
                    pick = rng.choice(good)
            pick = pick or (rng.choice(bad) if bad else rng.choice(chosen))
            for b in gate.buckets(pick):
                tally[i][b] += 1.0
        per_r[r] = st.mean(al)
    e = st.mean(per_r[r] for r in EARLY)
    l = st.mean(per_r[r] for r in LATE)
    return (l - e) / (1 - e)


def main():
    L = {c: load(c) for c in CONDS}
    have = sorted(set.intersection(*(set(L[c]) for c in CONDS)))
    print(f"model: {os.environ.get(chr(34)+chr(34)) or os.environ.get('AUDITREP_MODEL','upstage/solar-pro4')}")
    print(f"paired seeds: {have}  (n={len(have)})")
    if len(have) < 2:
        print("not enough runs yet")
        return
    S = {c: {s: shares(L[c][s]) for s in have} for c in CONDS}

    print(f"\n{'condition':10s} {'early':>7} {'late':>7} {'raw':>7} {'norm':>8} "
          f"{'adherence':>10} {'unparsed':>9}")
    for c in CONDS:
        f = lambda k: st.mean(S[c][s][k] for s in have)                  # noqa: E731
        print(f"{c:10s} {f('early'):7.3f} {f('late'):7.3f} {f('raw'):+7.3f} "
              f"{f('norm'):+8.3f} {f('adherence'):10.3f} "
              f"{sum(S[c][s]['unparsed'] for s in have):9d}")

    print("\nnormalised reinforcement per seed")
    print(f"{'seed':>5} " + "".join(f"{c:>12}" for c in CONDS))
    for s in have:
        print(f"{s:5d} " + "".join(f"{S[c][s]['norm']:+12.3f}" for c in CONDS))

    n = {c: [S[c][s]["norm"] for s in have] for c in CONDS}
    rng = random.Random(20260912)
    a1 = st.mean(n["pref_neg"]) > 0 and st.mean(n["pref_pos"]) > 0
    d2, p2 = perm_one_sided(n["pref_neg"], n["pref_pos"], rng)
    dh1, ph1 = perm_one_sided(n["pref_neg"], n["pref_h"], rng)
    dh2, ph2 = perm_one_sided(n["pref_pos"], n["pref_h"], rng)
    p3 = max(ph1, ph2)                       # A3 needs h below BOTH
    a3 = st.mean(n["pref_h"]) < min(st.mean(n["pref_neg"]), st.mean(n["pref_pos"]))
    ps = sorted([("A2", p2), ("A3", p3)], key=lambda t: t[1])
    holm = {}
    for i, (k, p) in enumerate(ps):
        holm[k] = min(1.0, p * (len(ps) - i))
    a2 = d2 > 0
    a4 = holm["A2"] < 0.05 and holm["A3"] < 0.05

    print(f"\nA1 reinforcement occurs (neg>0 and pos>0)        : "
          f"{'o' if a1 else 'x'}   neg {st.mean(n['pref_neg']):+.3f}  "
          f"pos {st.mean(n['pref_pos']):+.3f}")
    print(f"A2 negative stronger than positive               : "
          f"{'o' if a2 else 'x'}   diff {d2:+.3f}  p_raw {p2:.4f}  "
          f"p_holm {holm['A2']:.4f}")
    print(f"A3 meaningless preference below both             : "
          f"{'o' if a3 else 'x'}   h {st.mean(n['pref_h']):+.3f}  "
          f"p_raw {p3:.4f}  p_holm {holm['A3']:.4f}")
    print(f"A4 Holm-corrected one-sided p < 0.05 for A2, A3  : {'o' if a4 else 'x'}")

    import gate
    pool = gate.load_pool()
    ALIGN = {"pref_neg": gate.NEG, "pref_pos": gate.POS,
             "pref_h": {it["category"] for it in pool["items"] if it["high_h"]}}
    print("\nDIAGNOSTICS (not part of the criteria)")
    print(f"{'condition':10s} {'concentr':>9} {'adherence':>10} "
          f"{'observed':>9} {'matched null':>13} {'excess':>8}")
    diag = {}
    for c in CONDS:
        cc, dist = [], {}
        for s in have:
            k, d = concentration(L[c][s], ALIGN[c])
            cc.append(k)
            for kk, vv in d.items():
                dist[kk] = dist.get(kk, 0.0) + vv / len(have)
        adh = st.mean(S[c][s]["adherence"] for s in have)
        mn = st.mean(matched_null(c, s, pool, adh, dist if c != "pref_h" else {})
                     for s in have)
        obs = st.mean(n[c])
        diag[c] = {"concentration": st.mean(cc), "adherence": adh,
                   "observed": obs, "matched_null": mn, "excess": obs - mn,
                   "cat_dist": dist}
        print(f"{c:10s} {st.mean(cc):9.3f} {adh:10.3f} {obs:+9.3f} "
              f"{mn:+13.3f} {obs-mn:+8.3f}")
    print("  neg-pos gap: observed "
          f"{st.mean(n['pref_neg'])-st.mean(n['pref_pos']):+.3f}   "
          f"matched null {diag['pref_neg']['matched_null']-diag['pref_pos']['matched_null']:+.3f}")
    ok = a1 and a2 and a3 and a4
    print(f"\nCALIBRATION GATE: {'PASS' if ok else 'FAIL'}")

    json.dump({"seeds": have,
               "per_condition": {c: {str(s): S[c][s] for s in have} for c in CONDS},
               "norm": n, "A1": a1, "A2": a2, "A3": a3, "A4": a4,
               "diff_neg_pos": d2, "p_neg_pos": p2,
               "p_neg_h": ph1, "p_pos_h": ph2, "holm": holm, "PASS": ok, "diagnostics": diag},
              open(rundir().parent / f"verdict_gate_{rundir().name}.json", "w"), ensure_ascii=False, indent=1)
    print(f"wrote verdict_gate_{rundir().name}.json")


if __name__ == "__main__":
    main()
