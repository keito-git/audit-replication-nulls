"""A null agent calibrated to the real agents' measured response, used to check that
a proposed feed rule actually produces the intended treatment BEFORE spending.

The earlier null agent drifted to the feed mean, which is a far stronger and cleaner
response than the LLM agents actually have. Measured from the 50 completed runs:

    dominant band of the feed -> mean dv        sd
    neg   -0.043                                0.19
    neu   +0.017                                0.20
    pos   +0.021                                0.18
    separation pos-neg = +0.063

That is a signal-to-noise of about 0.33 per observation. Any rule that needs to
estimate a per-band response from ~4 observations must survive that.
"""
from __future__ import annotations

import random
import statistics as st

import sim2
from sim2 import band, new_learner, update_learner

SEP = {"neg": -0.043, "neu": +0.017, "pos": +0.021}
SD = 0.19


def select_rank(cond, items, lr, rng, topic_of, my_topic, vol=None):
    """Proposed repair: the platform RANKS bands by estimated response and admits the
    best ones, instead of testing each against an absolute zero. Real recommenders
    rank; an absolute threshold against a drifting, noisy estimate rejects everything
    roughly equally, which is what turned the treatment into a volume cut."""
    if cond != "affect_rank":
        return sim2.select(cond, items, lr, rng, topic_of, my_topic, vol)
    by_b = {b: [it for it in items if band(it[1]) == b] for b in sim2.BANDS}
    order = sorted(sim2.BANDS, key=lambda b: -lr["dv"][b])
    adm = order[:1]                      # the single best band: "only what helps you"
    if rng.random() < sim2.EPS:
        adm = adm + [rng.choice(sim2.BANDS)]
    cand = [it for b in set(adm) for it in by_b[b]]
    rng.shuffle(cand)
    return cand[:sim2.K_FEED]


def run(cond, seeds=20):
    vols, fv, m1s = [], [], []
    for seed in range(1, seeds + 1):
        rng0 = random.Random(seed)
        topic = [rng0.randrange(6) for _ in range(sim2.N_AGENTS)]
        v = [round(random.Random(seed * 7919 + i).uniform(-1, 1), 1)
             for i in range(sim2.N_AGENTS)]
        lrs = [new_learner() for _ in range(sim2.N_AGENTS)]
        for r in range(1, sim2.N_ROUNDS + 1):
            items = [(i, v[i], "") for i in range(sim2.N_AGENTS)]
            newv, vol_r = v[:], []
            for i in range(sim2.N_AGENTS):
                rng = random.Random(seed * 100003 + r * 997 + i)
                pool = [it for it in items if it[0] != i]
                f = select_rank(cond, pool, lrs[i], rng, topic, topic[i], sim2.K_FEED)
                vol_r.append(len(f))
                if f:
                    fv += [x[1] for x in f]
                    sh = {b: 0.0 for b in sim2.BANDS}
                    for x in f:
                        sh[band(x[1])] += 1.0 / len(f)
                    mu = sum(sh[b] * SEP[b] for b in sim2.BANDS)
                    dv = rng.gauss(mu, SD)
                else:
                    dv = rng.gauss(0.0, SD)
                newv[i] = max(-1.0, min(1.0, round(v[i] + dv, 2)))
                pick = rng.randrange(len(f)) if f else None
                update_learner("affect" if cond.startswith("affect") else cond,
                               lrs[i], f, pick, newv[i] - v[i])
            vols += vol_r
            v = newv
    return st.mean(vols), st.mean(fv)


print("null agent CALIBRATED to the measured response (separation +0.063, sd 0.19)\n")
print(f"{'rule':14s} {'mean volume':>12} {'mean feed valence':>19}")
for c in ["none", "affect", "affect_rank"]:
    vo, f = run(c)
    print(f"{c:14s} {vo:12.2f} {f:19.3f}")
print("\nobserved with real agents: none vol 6.00 feedV +0.016 / "
      "affect vol 4.20 feedV +0.016")
