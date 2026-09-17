"""Application-arm verdict. Applies the pre-registered rule mechanically.

Stage 1 is the calibration gate (V1-V3): the engagement objective must reproduce
the negative-emotion amplification that Habib & Nithyanand (arXiv:2501.15048)
audited on YouTube. If it does not, nothing is extrapolated to a regime that has
no audit to check against.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
SEEDS = [1, 2, 3, 4, 5]
N_ROUNDS, SHOCK_ROUND = 12, 9
PERM = 10000


def load(cond):
    out = {}
    for s in SEEDS:
        f = HERE / "runs" / f"{cond}_s{s}.json"
        if f.exists():
            out[s] = json.load(open(f, encoding="utf-8"))
    return out


def feed_vals(rd):
    """Feed valences with the common shock item removed (it is identical in every
    condition and is not a selector decision)."""
    fv = [list(x) for x in rd["feed_valences"]]
    if rd["r"] == SHOCK_ROUND:
        fv = [x[:-1] if x else x for x in fv]
    return fv


def m_val(log, lo=7, hi=12):
    vals = []
    for rd in log["rounds"]:
        if lo <= rd["r"] <= hi:
            flat = [v for f in feed_vals(rd) for v in f]
            if flat:
                vals.append(st.mean(flat))
    return st.mean(vals) if vals else float("nan")


def m_val_by_round(log):
    out = {}
    for rd in log["rounds"]:
        flat = [v for f in feed_vals(rd) for v in f]
        out[rd["r"]] = st.mean(flat) if flat else float("nan")
    return out


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def m1_inequality(log):
    rs = []
    for rd in log["rounds"]:
        r = spearman(rd["v_old"], rd["volumes"])
        if r == r:
            rs.append(r)
    return st.mean(rs) if rs else float("nan")


def m2_trap(log, lag=4):
    hits = tot = 0
    rounds = {rd["r"]: rd["v"] for rd in log["rounds"]}
    for r in range(1, N_ROUNDS - lag + 1):
        v0, v1 = rounds[r], rounds[r + lag]
        q0 = sorted(v0)[len(v0) // 4]
        q1 = sorted(v1)[len(v1) // 4]
        low = [i for i, x in enumerate(v0) if x <= q0]
        for i in low:
            tot += 1
            hits += v1[i] <= q1
    return hits / tot if tot else float("nan")


def m3_legibility(log, at=12):
    for rd in log["rounds"]:
        if rd["r"] == at and "attribution" in rd:
            a = rd["attribution"]
            errs = [abs(g - t) for g, t in zip(a["guess"], a["truth"]) if g is not None]
            return st.mean(errs) if errs else float("nan")
    return float("nan")


def m4_tolerance(log):
    for rd in log["rounds"]:
        if rd["r"] == SHOCK_ROUND:
            return st.mean(abs(a - b) for a, b in zip(rd["v"], rd["v_old"]))
    return float("nan")


def perm_two(a, b, rng):
    """Seed-blocked permutation: swap the two condition labels within each seed."""
    obs = st.mean(a) - st.mean(b)
    ge = 0
    for _ in range(PERM):
        d = []
        for x, y in zip(a, b):
            d.append(x - y if rng.random() < 0.5 else y - x)
        if abs(st.mean(d)) >= abs(obs) - 1e-12:
            ge += 1
    return obs, (ge + 1) / (PERM + 1)


def stage1():
    none, eng = load("none"), load("engagement")
    have = sorted(set(none) & set(eng))
    print(f"calibration gate on {len(have)} paired seeds: {have}\n")
    if not have:
        print("no paired runs yet")
        return
    a = [m_val(eng[s]) for s in have]
    b = [m_val(none[s]) for s in have]
    print(f"{'seed':>5} {'engagement':>12} {'none':>10} {'gap':>8}")
    for s, x, y in zip(have, a, b):
        print(f"{s:5d} {x:12.3f} {y:10.3f} {x-y:+8.3f}")
    obs, p = perm_two(a, b, random.Random(20260912))
    print(f"\nV1 mean valence of received items, rounds 7-12")
    print(f"   engagement {st.mean(a):+.3f}   none {st.mean(b):+.3f}   "
          f"gap {obs:+.3f}   (needs <= -0.10)")
    gaps = []
    for s in have:
        pe, pn = m_val_by_round(eng[s]), m_val_by_round(none[s])
        gaps.append({r: pe[r] - pn[r] for r in range(1, N_ROUNDS + 1)})
    per_round = {r: st.mean(g[r] for g in gaps) for r in range(1, N_ROUNDS + 1)}
    trend = pearson(list(range(2, N_ROUNDS + 1)),
                    [per_round[r] for r in range(2, N_ROUNDS + 1)])
    print("\nV2 gap by round (engagement - none):")
    print("   " + "  ".join(f"r{r}:{per_round[r]:+.2f}" for r in range(1, N_ROUNDS + 1)))
    print(f"   trend corr(round, gap) = {trend:+.3f}   (needs < 0, i.e. widening "
          f"in the negative direction)")
    print(f"\nV3 seed-blocked permutation p = {p:.4f}   (needs < 0.05)")
    v1, v2, v3 = obs <= -0.10, trend < 0, p < 0.05
    print(f"\nV1 {'o' if v1 else 'x'}  V2 {'o' if v2 else 'x'}  V3 {'o' if v3 else 'x'}"
          f"  =>  {'PASS' if (v1 and v2 and v3) else 'FAIL'}")
    json.dump({"stage": 1, "seeds": have,
               "m_val_engagement": a, "m_val_none": b, "gap": obs,
               "gap_by_round": per_round, "trend": trend, "p": p,
               "V1": v1, "V2": v2, "V3": v3, "PASS": bool(v1 and v2 and v3)},
              open(HERE / "verdict_stage1.json", "w"), ensure_ascii=False, indent=1)
    print("\nwrote verdict_stage1.json")


def describe(cond):
    lg = load(cond)
    if not lg:
        print(f"{cond}: no runs")
        return
    rows = {"M_val": [], "M1_ineq": [], "M2_trap": [], "M3_MAE": [], "M4_shock": [],
            "mean_vol": [], "final_mean_v": [], "parse_fail": []}
    for s in sorted(lg):
        L = lg[s]
        rows["M_val"].append(m_val(L))
        rows["M1_ineq"].append(m1_inequality(L))
        rows["M2_trap"].append(m2_trap(L))
        rows["M3_MAE"].append(m3_legibility(L))
        rows["M4_shock"].append(m4_tolerance(L))
        rows["mean_vol"].append(st.mean(v for rd in L["rounds"] for v in rd["volumes"]))
        rows["final_mean_v"].append(st.mean(L["rounds"][-1]["v"]))
        rows["parse_fail"].append(L["parse_failures"])
    print(f"\n[{cond}]  n={len(lg)} seeds")
    for k, v in rows.items():
        vv = [x for x in v if x == x]
        if vv:
            print(f"  {k:14s} {st.mean(vv):+8.3f}  sd {st.pstdev(vv):.3f}   {[round(x,3) for x in v]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stage1":
        stage1()
    else:
        for c in (sys.argv[1:] or ["none", "engagement", "relevance", "affect",
                                   "volume_null"]):
            describe(c)
