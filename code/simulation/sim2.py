"""Application arm: a society under feeds that optimise different objectives.

Built on the machinery the calibration gate validated (tally over bands, learned
from choices), so the warrant the gate earned carries into this arm.

Every condition runs the identical procedure: a feed is delivered, the agent picks
one item to open, then the agent updates its state and posts. Only the objective
that selects the feed differs, and only which signal the learner consumes:

  none          random feed, nothing learned
  relevance     topical match, nothing learned
  engagement    learns from the CHOICE            (a behavioural signal)
  affect        learns from the measured Delta v  (requires measuring feeling)
  volume_null   random feed at exactly the volume `affect` delivered

The pick call is made in every condition so that the procedure is identical; only
`engagement` uses it. It never enters the state-update prompt.

Usage:
  python3 sim2.py check                 null-agent test of the selector, no LLM calls
  python3 sim2.py run affect            stage A
  python3 sim2.py run volume_null ...   after extract_volumes.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import pathlib
import random
import re
import statistics as st
import sys

import sim

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
OUT = HERE / "runs2"
OUT.mkdir(exist_ok=True)

N_AGENTS, N_ROUNDS, K_FEED = 24, 12, 6
SEEDS = list(range(1, 11))
EPS = 0.2
SHOCK_ROUND = 9
SHOCK = {"valence": -0.9,
         "text": "They have started tearing it down and no one will say when it comes back."}
BANDS = ["neg", "neu", "pos"]
CONDS = ["none", "relevance", "engagement", "affect", "volume_null"]
ATTR_ROUNDS = (1, 6, 12)


def band(v):
    return "neg" if v < -0.2 else ("pos" if v > 0.2 else "neu")


SYS_PICK = "You are one person browsing a feed. You answer only with a number."
TMPL_PICK = ("{persona}\nYou care most about {topic}.\n\n"
             "Right now your feeling is {v:+.1f} on a scale from -1.0 (very bad) to "
             "+1.0 (very good).\n\n"
             "Your feed shows these posts:\n{slate}\n\n"
             "Answer with exactly one number from 1 to {n}: which post do you open?\n"
             "Answer:")


def select(cond, items, learner, rng, topic_of, my_topic, vol=None, my_v=0.0):
    """items: list of (author, valence, text). Returns the delivered feed."""
    if cond == "none":
        return rng.sample(items, min(K_FEED, len(items)))
    if cond == "volume_null":
        return rng.sample(items, min(int(vol), len(items)))
    if cond == "relevance":
        return sorted(items, key=lambda it: (-(topic_of[it[0]] == my_topic),
                                             rng.random()))[:K_FEED]
    by_b = {b: [it for it in items if band(it[1]) == b] for b in BANDS}
    if cond == "engagement":
        # the machinery the gate validated: sample proportional to the tally
        chosen, cand = [], items[:]
        n_rand = sum(1 for _ in range(K_FEED) if rng.random() < EPS)
        rng.shuffle(cand)
        chosen += cand[:n_rand]
        ids = {id(x) for x in chosen}
        cand = [it for it in items if id(it) not in ids]
        while len(chosen) < K_FEED and cand:
            w = [1.0 + learner["tally"][band(it[1])] for it in cand]
            x = rng.random() * sum(w)
            acc = 0.0
            for it, wi in zip(cand, w):
                acc += wi
                if acc >= x:
                    chosen.append(it)
                    cand.remove(it)
                    break
        return chosen
    if cond == "affect":
        if isinstance(learner, PooledAffect):
            adm = learner.admissible(band(my_v))
        else:
            adm = [b for b in BANDS if learner["dv"][b] >= 0.0]
        if rng.random() < EPS:
            extra = rng.choice(BANDS)
            if extra not in adm:
                adm.append(extra)
        cand = [it for b in adm for it in by_b[b]]
        rng.shuffle(cand)
        return cand[:K_FEED]          # may be fewer: that is the treatment
    raise ValueError(cond)


class PooledAffect:
    """The platform learns across all users, not per user.

    Repaired 2026-09-12. A per-agent learner gets about four observations per band in
    twelve rounds, and the measured response separates by only +0.063 against a
    per-observation sd of 0.19, so its estimate sits below the noise floor and the
    admissibility test rejected every band roughly equally -- the treatment became a
    uniform volume cut with no content selection. Pooling over 24 agents x 12 rounds
    gives about 32 observations per (recipient band, item band) cell. Real recommender
    systems pool over millions of users; the per-agent version was the unrealistic one.

    Conditioning on the recipient's own band is required, not optional: without it
    every agent receives the same admissible set, volume is constant across agents,
    and K1 cannot be computed at all.
    """

    def __init__(self):
        self.m = {(rb, ib): 0.0 for rb in BANDS for ib in BANDS}
        self.w = {(rb, ib): 1.0 for rb in BANDS for ib in BANDS}

    def update(self, recipient_band, shares, dv):
        for ib, s in shares.items():
            if s <= 0:
                continue
            k = (recipient_band, ib)
            self.w[k] += s
            self.m[k] += s * (dv - self.m[k]) / self.w[k]

    def admissible(self, recipient_band):
        adm = [ib for ib in BANDS if self.m[(recipient_band, ib)] >= 0.0]
        return adm or [max(BANDS, key=lambda ib: self.m[(recipient_band, ib)])]


def new_learner():
    return {"tally": {b: 0.0 for b in BANDS},
            # neutral, not optimistic. With an optimistic +1.0 the admissibility test
            # decays too slowly to ever bind and `affect` behaves exactly like `none`;
            # the null-agent test caught that before any LLM call was made.
            "dv": {b: 0.0 for b in BANDS},
            "w": {b: 1.0 for b in BANDS}}


def update_learner(cond, lr, feed, pick_idx, dv, recipient_band=None):
    if cond == "affect" and isinstance(lr, PooledAffect) and feed:
        sh = {b: 0.0 for b in BANDS}
        for (_, iv, _) in feed:
            sh[band(iv)] += 1.0 / len(feed)
        lr.update(recipient_band, sh, dv)
        return
    if cond == "engagement":
        if pick_idx is not None:
            lr["tally"][band(feed[pick_idx][1])] += 1.0
    elif cond == "affect" and feed:
        sh = {b: 0.0 for b in BANDS}
        for (_, iv, _) in feed:
            sh[band(iv)] += 1.0 / len(feed)
        for b, s in sh.items():
            if s > 0:
                lr["w"][b] += s
                lr["dv"][b] += s * (dv - lr["dv"][b]) / lr["w"][b]


# ------------------------------------------------------ null-agent selector test
def check_selector():
    """Run the selector with the LLM replaced by a fixed random agent, so any
    difference between conditions is a property of the selector alone. This is the
    check that the calibration gate's first failure taught us to run first."""
    print("null-agent selector test (no LLM calls)\n")
    print(f"{'condition':12s} {'mean volume':>12} {'mean feed valence':>19} "
          f"{'share neg':>10} {'share pos':>10}")
    rows = {}
    for cond in ["none", "relevance", "engagement", "affect"]:
        vols, fv, sn, sp = [], [], [], []
        for seed in range(1, 21):
            rng0 = random.Random(seed)
            topic_of = [rng0.randrange(6) for _ in range(N_AGENTS)]
            v = [round(random.Random(seed * 7919 + i).uniform(-1, 1), 1)
                 for i in range(N_AGENTS)]
            lrs = [new_learner() for _ in range(N_AGENTS)]
            for r in range(1, N_ROUNDS + 1):
                items = [(i, v[i], "") for i in range(N_AGENTS)]
                newv = v[:]
                for i in range(N_AGENTS):
                    rng = random.Random(seed * 100003 + r * 997 + i)
                    pool = [it for it in items if it[0] != i]
                    f = select(cond, pool, lrs[i], rng, topic_of, topic_of[i], K_FEED)
                    vols.append(len(f))
                    if f:
                        fv += [x[1] for x in f]
                        sn.append(sum(1 for x in f if band(x[1]) == "neg") / len(f))
                        sp.append(sum(1 for x in f if band(x[1]) == "pos") / len(f))
                    # the null agent: drifts toward the mean of what it was shown
                    tgt = st.mean(x[1] for x in f) if f else v[i]
                    newv[i] = max(-1.0, min(1.0, round(v[i] + 0.3 * (tgt - v[i]), 2)))
                    pick = rng.randrange(len(f)) if f else None
                    update_learner(cond, lrs[i], f, pick, newv[i] - v[i])
                v = newv
        rows[cond] = (st.mean(vols), st.mean(fv), st.mean(sn), st.mean(sp))
        print(f"{cond:12s} {rows[cond][0]:12.2f} {rows[cond][1]:19.3f} "
              f"{rows[cond][2]:10.3f} {rows[cond][3]:10.3f}")
    print("\nWhat this must show before any spend:")
    print("  - none and relevance: volume 6.00 and feed valence near the pool mean")
    print("  - engagement: volume 6.00 (it never withholds)")
    print("  - affect: volume BELOW 6 and feed valence ABOVE the others")
    print("    (the treatment is a constraint, so reduced volume is intrinsic, not a bug)")
    return rows


# ------------------------------------------------------------------------ run
def run_one(cond, seed, vols=None):
    pers = [sim.persona(random.Random(seed * 1000 + i)) for i in range(N_AGENTS)]
    topic_of = [sim.TOPICS[random.Random(seed * 77 + i).randrange(len(sim.TOPICS))]
                for i in range(N_AGENTS)]
    v = [round(random.Random(seed * 7919 + i).uniform(-1, 1), 1) for i in range(N_AGENTS)]
    pooled = PooledAffect() if cond == "affect" else None
    lrs = [pooled if pooled else new_learner() for _ in range(N_AGENTS)]
    parse_fail = 0

    def update_call(i, feed_txt, r):
        msg = [{"role": "system", "content": sim.SYS},
               {"role": "user", "content": sim.TMPL.format(persona=pers[i],
                                                           topic=topic_of[i], v=v[i],
                                                           feed=feed_txt)}]
        return i, sim.chat(msg, seed=seed * 31 + r)

    posts = [None] * N_AGENTS
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for i, txt in ex.map(lambda i: update_call(i, "nothing", 0), range(N_AGENTS)):
            nv, p = sim.parse(txt, v[i])
            if nv is None or not p:
                parse_fail += 1
            if nv is not None:
                v[i] = nv
            posts[i] = p or "Nothing much to say today."
    items = [(i, v[i], posts[i]) for i in range(N_AGENTS)]

    log = {"condition": cond, "seed": seed, "model": sim.MODEL, "topic_of": topic_of,
           "round0": {"v": v[:], "posts": posts[:]}, "rounds": []}

    for r in range(1, N_ROUNDS + 1):
        feeds = {}
        for i in range(N_AGENTS):
            pool = [it for it in items if it[0] != i]
            rng = random.Random(seed * 100003 + r * 997 + i)
            vol = vols[str(seed)][str(r)][i] if vols else None
            f = select(cond, pool, lrs[i], rng, topic_of, topic_of[i], vol, v[i])
            if r == SHOCK_ROUND:
                f = f + [(-1, SHOCK["valence"], SHOCK["text"])]
            feeds[i] = f

        # the pick call is made in EVERY condition so the procedure is identical;
        # only `engagement` consumes it, and it never enters the update prompt
        picks = [None] * N_AGENTS

        def pick_call(i):
            f = feeds[i]
            if not f:
                return i, None
            sl = "\n".join(f"{j+1}. {t}" for j, (_, _, t) in enumerate(f))
            m = [{"role": "system", "content": SYS_PICK},
                 {"role": "user", "content": TMPL_PICK.format(persona=pers[i],
                                                              topic=topic_of[i], v=v[i],
                                                              slate=sl, n=len(f))}]
            txt = sim.chat(m, max_tokens=6, seed=seed * 17 + r)
            n = re.findall(r"\d+", txt or "")
            k = int(n[0]) - 1 if n else -1
            return i, (k if 0 <= k < len(f) else None)

        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            for i, k in ex.map(pick_call, range(N_AGENTS)):
                picks[i] = k

        v_old = v[:]
        txts = [None] * N_AGENTS
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            def work(i):
                ft = "\n".join(f"- {t}" for (_, _, t) in feeds[i]) or "nothing"
                return update_call(i, ft, r)
            for i, txt in ex.map(work, range(N_AGENTS)):
                txts[i] = txt
        for i in range(N_AGENTS):
            nv, p = sim.parse(txts[i], v[i])
            if nv is None or not p:
                parse_fail += 1
            if nv is not None:
                v[i] = nv
            if p:
                posts[i] = p

        for i in range(N_AGENTS):
            update_learner(cond, lrs[i], feeds[i], picks[i], v[i] - v_old[i],
                           band(v_old[i]))

        rd = {"r": r, "v": v[:], "v_old": v_old, "posts": posts[:],
              "volumes": [len([x for x in feeds[i] if x[0] != -1]) for i in range(N_AGENTS)],
              "feed_valences": [[x[1] for x in feeds[i]] for i in range(N_AGENTS)],
              "picks": picks,
              "pick_valence": [None if picks[i] is None else feeds[i][picks[i]][1]
                               for i in range(N_AGENTS)],
              "bands": ({f"{a}|{b}": pooled.m[(a, b)] for a in BANDS for b in BANDS}
                        if pooled else
                        [dict(lrs[i]["tally"]) for i in range(N_AGENTS)])}

        if r in ATTR_ROUNDS:
            tgt = [(i + 1 + random.Random(seed * 13 + r).randrange(N_AGENTS - 1)) % N_AGENTS
                   for i in range(N_AGENTS)]
            att = [None] * N_AGENTS
            with cf.ThreadPoolExecutor(max_workers=12) as ex:
                def attr(i):
                    m = [{"role": "system", "content": sim.SYS_ATTR},
                         {"role": "user",
                          "content": sim.TMPL_ATTR.format(post=posts[tgt[i]])}]
                    return i, sim.chat(m, max_tokens=8, seed=seed * 7 + r)
                for i, tx in ex.map(attr, range(N_AGENTS)):
                    n = re.findall(r"[-+]?\d*\.?\d+", tx or "")
                    att[i] = max(-1.0, min(1.0, float(n[0]))) if n else None
            rd["attribution"] = {"target": tgt, "guess": att, "truth": [v[j] for j in tgt]}

        log["rounds"].append(rd)
        items = [(i, v[i], posts[i]) for i in range(N_AGENTS)]
        print(f"  {cond} s{seed} r{r}: mean v {st.mean(v):+.3f}  "
              f"mean vol {st.mean(rd['volumes']):.2f}", flush=True)

    log["parse_failures"] = parse_fail
    return log


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check_selector()
    else:
        vols = None
        vf = HERE / "affect_volumes.json"
        if "volume_null" in sys.argv[2:]:
            if not vf.exists():
                raise SystemExit("run extract_volumes.py after the affect runs first")
            vols = json.load(open(vf))
        for cond in (sys.argv[2:] or CONDS):
            for s in SEEDS:
                f = OUT / f"{cond}_s{s}.json"
                if f.exists():
                    print(f"{cond} s{s}: present")
                    continue
                lg = run_one(cond, s, vols)
                f.write_text(json.dumps(lg, ensure_ascii=False))
                print(f"{cond} s{s}: done, parse failures {lg['parse_failures']}",
                      flush=True)
