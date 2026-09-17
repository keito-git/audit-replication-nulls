"""Verdict for the application arm. Applies §4 stage 2 (K1-K4) with the 2026-09-12 amendments:
K1 is two-sided with the sign reported, and every mechanically-reproducible measure
carries a null-agent baseline so nothing is compared against zero.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import statistics as st

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
CONDS = ["none", "relevance", "engagement", "affect", "volume_null"]
SEEDS = list(range(1, 11))
N_ROUNDS, SHOCK_ROUND = 12, 9
PERM = 10000


def load(cond):
    out = {}
    for s in SEEDS:
        f = HERE / "runs2" / f"{cond}_s{s}.json"
        if f.exists():
            out[s] = json.load(open(f, encoding="utf-8"))
    return out


def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[o[k]] = a
        i = j + 1
    return r


def pearson(a, b):
    if len(a) < 3:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def feed_vals(rd):
    fv = [list(x) for x in rd["feed_valences"]]
    if rd["r"] == SHOCK_ROUND:
        fv = [x[:-1] if x else x for x in fv]
    return fv


def M1(log):
    """rho(state facing the selector, volume delivered). Two-sided; sign reported."""
    rs = [spearman(rd["v_old"], rd["volumes"]) for rd in log["rounds"]
          if len(set(rd["volumes"])) > 1]
    return st.mean(rs) if rs else float("nan")


def M2(log, lag=4):
    rounds = {rd["r"]: rd["v"] for rd in log["rounds"]}
    hits = tot = 0
    for r in range(1, N_ROUNDS - lag + 1):
        v0, v1 = rounds[r], rounds[r + lag]
        q0, q1 = sorted(v0)[len(v0) // 4], sorted(v1)[len(v1) // 4]
        for i, x in enumerate(v0):
            if x <= q0:
                tot += 1
                hits += v1[i] <= q1
    return hits / tot if tot else float("nan")


def M3(log, at=12):
    for rd in log["rounds"]:
        if rd["r"] == at and "attribution" in rd:
            a = rd["attribution"]
            e = [abs(g - t) for g, t in zip(a["guess"], a["truth"]) if g is not None]
            return st.mean(e) if e else float("nan")
    return float("nan")


def M4(log):
    for rd in log["rounds"]:
        if rd["r"] == SHOCK_ROUND:
            return st.mean(abs(a - b) for a, b in zip(rd["v"], rd["v_old"]))
    return float("nan")


def feed_mean(log):
    v = [x for rd in log["rounds"] for f in feed_vals(rd) for x in f]
    return st.mean(v) if v else float("nan")


def vol_mean(log):
    return st.mean(x for rd in log["rounds"] for x in rd["volumes"])


def perm_paired(a, b, rng, one_sided=True):
    d = [x - y for x, y in zip(a, b)]
    obs = st.mean(d)
    ge = 0
    for _ in range(PERM):
        f = [x if rng.random() < 0.5 else -x for x in d]
        m = st.mean(f)
        if (m >= obs - 1e-12) if one_sided else (abs(m) >= abs(obs) - 1e-12):
            ge += 1
    return obs, (ge + 1) / (PERM + 1)


def null_baselines():
    """Baselines from a null agent CALIBRATED to the measured response (separation
    +0.063 between the pos and neg feed bands, per-observation sd 0.19), running the
    same selectors -- including the pooled affect learner. Comparing an observed M1 or
    M2 against zero would credit the agents with whatever the rule produces mechanically.
    """
    import sim2
    SEP = {"neg": -0.043, "neu": +0.017, "pos": +0.021}
    SD = 0.19
    out = {}
    for cond in ["none", "relevance", "engagement", "affect"]:
        m1s, m2s, fvs, vols = [], [], [], []
        for seed in range(1, 21):
            rng0 = random.Random(seed)
            topic = [rng0.randrange(6) for _ in range(sim2.N_AGENTS)]
            v = [round(random.Random(seed * 7919 + i).uniform(-1, 1), 1)
                 for i in range(sim2.N_AGENTS)]
            pooled = sim2.PooledAffect() if cond == "affect" else None
            lrs = [pooled if pooled else sim2.new_learner()
                   for _ in range(sim2.N_AGENTS)]
            hist = {}
            for r in range(1, sim2.N_ROUNDS + 1):
                items = [(i, v[i], "") for i in range(sim2.N_AGENTS)]
                newv, vol_r = v[:], []
                for i in range(sim2.N_AGENTS):
                    rng = random.Random(seed * 100003 + r * 997 + i)
                    pool = [it for it in items if it[0] != i]
                    f = sim2.select(cond, pool, lrs[i], rng, topic, topic[i],
                                    sim2.K_FEED, v[i])
                    vol_r.append(len(f))
                    if f:
                        fvs += [x[1] for x in f]
                        sh = {b: 0.0 for b in sim2.BANDS}
                        for x in f:
                            sh[sim2.band(x[1])] += 1.0 / len(f)
                        dv = rng.gauss(sum(sh[b] * SEP[b] for b in sim2.BANDS), SD)
                    else:
                        dv = rng.gauss(0.0, SD)
                    newv[i] = max(-1.0, min(1.0, round(v[i] + dv, 2)))
                    pick = rng.randrange(len(f)) if f else None
                    sim2.update_learner(cond, lrs[i], f, pick, newv[i] - v[i],
                                        sim2.band(v[i]))
                if len(set(vol_r)) > 1:
                    m1s.append(spearman(v, vol_r))
                vols += vol_r
                hist[r] = newv[:]
                v = newv
            hits = tot = 0
            for r in range(1, sim2.N_ROUNDS - 4 + 1):
                v0, v1 = hist[r], hist[r + 4]
                q0, q1 = sorted(v0)[len(v0) // 4], sorted(v1)[len(v1) // 4]
                for i, x in enumerate(v0):
                    if x <= q0:
                        tot += 1
                        hits += v1[i] <= q1
            m2s.append(hits / tot if tot else float("nan"))
        out[cond] = {"M1": st.mean(m1s) if m1s else float("nan"),
                     "M2": st.mean(m2s), "feed_mean": st.mean(fvs),
                     "vol": st.mean(vols)}
    return out


def main():
    L = {c: load(c) for c in CONDS}
    for c in CONDS:
        print(f"{c:12s} {len(L[c])}/10 runs")
    have = sorted(set.intersection(*(set(L[c]) for c in CONDS))) if all(L.values()) else []
    if len(have) < 5:
        print("\nnot enough paired runs yet")
        return
    print(f"\npaired seeds: {have} (n={len(have)})\n")

    tab = {c: {"M1": [M1(L[c][s]) for s in have],
               "M2": [M2(L[c][s]) for s in have],
               "M3": [M3(L[c][s]) for s in have],
               "M4": [M4(L[c][s]) for s in have],
               "feed_mean": [feed_mean(L[c][s]) for s in have],
               "vol": [vol_mean(L[c][s]) for s in have],
               "pf": [L[c][s]["parse_failures"] for s in have]} for c in CONDS}

    print(f"{'condition':12s} {'vol':>6} {'feedV':>7} {'M1 rho':>8} {'M2 trap':>8} "
          f"{'M3 MAE':>8} {'M4 shock':>9} {'parsefail':>10}")
    for c in CONDS:
        def f(k, c=c):
            xs = [x for x in tab[c][k] if x == x]
            return st.mean(xs) if xs else float("nan")
        m1 = f("M1")
        m1s = f"{m1:+8.3f}" if m1 == m1 else "     n/a"   # undefined at constant volume
        print(f"{c:12s} {f('vol'):6.2f} {f('feed_mean'):+7.3f} {m1s} "
              f"{f('M2'):8.3f} {f('M3'):8.3f} {f('M4'):9.3f} {sum(tab[c]['pf']):10d}")

    print("\nnull-agent baselines (same selector, LLM replaced by a drift agent)")
    nb = null_baselines()
    print(f"{'condition':12s} {'vol':>6} {'feedV':>7} {'M1 rho':>8} {'M2 trap':>8}")
    for c in nb:
        r = nb[c]["M1"]
        rs = f"{r:+8.3f}" if r == r else "     n/a"
        print(f"{c:12s} {nb[c]['vol']:6.2f} {nb[c]['feed_mean']:+7.3f} {rs} "
              f"{nb[c]['M2']:8.3f}")

    rng = random.Random(20260912)
    res = {}

    # K1: two-sided, within `affect`, sign reported, null-agent baseline alongside
    _a = [x for x in tab["affect"]["M1"] if x == x]
    r_aff = st.mean(_a)
    k1 = abs(r_aff) >= 0.30
    _, p1 = perm_paired(_a, [0.0] * len(_a), rng, one_sided=False)
    res["K1"] = {"rho": r_aff, "sign": "positive" if r_aff > 0 else "negative",
                 "abs_ge_030": k1, "p": p1, "null_agent_rho": nb["affect"]["M1"],
                 "excess_over_null": r_aff - nb["affect"]["M1"]}

    # K2-K4: affect vs none, must also exceed half the gap against volume_null
    raw_p = {}
    for key, m, thr, direction in (("K2", "M3", 0.15, +1), ("K3", "M2", 0.15, +1),
                                   ("K4", "M4", 0.15, +1)):
        a = tab["affect"][m]
        b = tab["none"][m]
        vn = tab["volume_null"][m]
        d_none = st.mean(a) - st.mean(b)
        d_vnull = st.mean(a) - st.mean(vn)
        obs, p = perm_paired(a, b, rng, one_sided=True)
        raw_p[key] = p
        res[key] = {"measure": m, "affect": st.mean(a), "none": st.mean(b),
                    "volume_null": st.mean(vn),
                    "gap_vs_none": d_none, "gap_vs_volume_null": d_vnull,
                    "c_effect": d_none >= thr,
                    "c_not_volume": (d_vnull >= 0.5 * d_none) if d_none > 0 else False,
                    "p_raw": p}
    order = sorted(raw_p, key=lambda k: raw_p[k])
    for i, k in enumerate(order):
        res[k]["p_holm"] = min(1.0, raw_p[k] * (len(order) - i))
        res[k]["PASS"] = (res[k]["c_effect"] and res[k]["c_not_volume"]
                          and res[k]["p_holm"] < 0.05)

    print("\nPRE-REGISTERED CRITERIA")
    print(f"K1 |rho| >= 0.30 in `affect`, sign reported")
    print(f"   rho {r_aff:+.3f} ({res['K1']['sign']})   |rho|>=0.30 "
          f"{'o' if k1 else 'x'}   p(two-sided) {p1:.4f}")
    print(f"   null-agent rho {nb['affect']['M1']:+.3f}   "
          f"excess over null {r_aff - nb['affect']['M1']:+.3f}")
    for k in ("K2", "K3", "K4"):
        r = res[k]
        print(f"{k} ({r['measure']}): affect {r['affect']:.3f}  none {r['none']:.3f}  "
              f"vol_null {r['volume_null']:.3f}  gap {r['gap_vs_none']:+.3f}  "
              f"p_holm {r['p_holm']:.4f}  "
              f"[effect {'o' if r['c_effect'] else 'x'} "
              f"beyond-volume {'o' if r['c_not_volume'] else 'x'}] "
              f"=> {'PASS' if r['PASS'] else 'FAIL'}")
    n_pass = sum(res[k]["PASS"] for k in ("K2", "K3", "K4")) + (1 if k1 else 0)
    overall = n_pass >= 2
    print(f"\n{n_pass} of K1-K4 passed; PASS needs >= 2  =>  "
          f"{'PASS' if overall else 'FAIL'}")

    sat = {c: st.mean(tab[c]["M2"]) for c in CONDS}
    print(f"\nsaturation check on M2: {min(sat.values()):.3f}-{max(sat.values()):.3f}")

    json.dump({"seeds": have, "table": {c: {k: v for k, v in tab[c].items()}
                                        for c in CONDS},
               "null_baselines": nb, "criteria": res, "n_pass": n_pass,
               "PASS": overall},
              open(HERE / "verdict2.json", "w"), ensure_ascii=False, indent=1)
    print("wrote verdict2.json")


if __name__ == "__main__":
    main()
