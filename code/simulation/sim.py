"""Application arm: what does a feed optimised for the recipient's affect do to a society?

Nothing here may be changed after the first full run without dating the change.

The only thing that differs between conditions is the objective the feed selector
optimises. The learner, the prompt, the personas, the graph-free platform, the
rounds, the model, the temperature and the seeds are identical; assert_parity()
checks the prompt mechanically before any call is made.

Agents see the TEXT of the posts selected for them. The selector sees only the
declared valence numbers. That separation is deliberate: the agent reacts to
language, the platform optimises a number.

Usage:
  python3 sim.py smoke                  none, seed 1
  python3 sim.py run none engagement    the calibration gate
  python3 sim.py run affect relevance   stage 2 (volume_null needs affect first)
  python3 sim.py run volume_null        replays the volumes affect delivered
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import ssl
import time
import sys
import urllib.request

import certifi

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
OUT = HERE / "runs"
OUT.mkdir(exist_ok=True)
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "upstage/solar-pro4"
TEMP = 1.0

N_AGENTS, N_ROUNDS, K_FEED = 24, 12, 6
SEEDS = [1, 2, 3, 4, 5]
EPS = 0.2
SHOCK_ROUND = 9
SHOCK = {"valence": -0.9,
         "text": "They have started tearing it down and no one will say when it comes back."}

BIG5 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TOPICS = ["the neighbourhood", "work", "family", "sport", "music", "food"]
CONDITIONS = ["none", "engagement", "relevance", "affect", "volume_null"]
BUCKETS = ["neg", "neu", "pos"]


def bucket(v):
    return "neg" if v < -0.2 else ("pos" if v > 0.2 else "neu")


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY in the environment")
    return key


# resolved on first use, so that analyses which only read stored runs
# need no credentials
KEY = None


def chat(messages, max_tokens=60, seed=0, retries=8):
    """One call, with exponential backoff.

    A long confirmatory sweep meets the provider's rate limit sooner or later.
    Four immediate retries were not enough: a sweep died at seed 39 on HTTP 429
    and the remaining sixty seeds were lost. Backing off costs nothing when the
    service is healthy and keeps a multi-hour sweep from losing its place.
    """
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": TEMP,
                       "max_tokens": max_tokens, "seed": seed}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
                d = json.load(r)
            return (d["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:                                   # noqa: BLE001
            last = e
            code = getattr(e, "code", None)
            time.sleep(min(60.0, (2 ** attempt) * (2.0 if code == 429 else 0.5)))
    raise RuntimeError(f"chat failed after {retries}: {last}")


# ------------------------------------------------------------------- prompting
SYS = ("You are one person using a social feed, thinking privately. "
       "You answer only in the format asked for, with no explanation.")
TMPL = ("{persona}\nYou care most about {topic}.\n\n"
        "Right now your feeling is {v:+.1f} on a scale from -1.0 (very bad) to "
        "+1.0 (very good).\n\n"
        "Your feed shows you:\n{feed}\n\n"
        "Having read your feed, answer in exactly this format and nothing else:\n"
        "VALENCE: <a number between -1.0 and 1.0>\n"
        "POST: <one sentence, at most 20 words, that you would post now>")

SYS_ATTR = "You are reading one post from a stranger. You answer only with a number."
TMPL_ATTR = ('Someone posted this:\n"{post}"\n\n'
             "Answer with exactly one number between -1.0 and 1.0: how do you think "
             "the writer feels? -1.0 is very bad, +1.0 is very good.\nAnswer:")


def persona(rng):
    t = {b: rng.choice(["low", "moderate", "high"]) for b in BIG5}
    return "You are a person with " + ", ".join(f"{v} {b}" for b, v in t.items()) + "."


def assert_parity():
    """The prompt must be byte-identical across conditions outside the feed slot."""
    p = persona(random.Random(0))
    holes = {}
    for c in CONDITIONS:
        msg = TMPL.format(persona=p, topic=TOPICS[0], v=0.3, feed="<FEED>")
        holes[c] = msg
    ref = next(iter(holes.values()))
    for c, m in holes.items():
        if m != ref:
            raise AssertionError(f"prompt differs outside the feed slot: {c}")
    print(f"prompt parity: {len(holes)} conditions identical outside the feed slot")


def parse(txt, v_old):
    mv = re.search(r"VALENCE\s*:\s*([-+]?\d*\.?\d+)", txt or "", re.I)
    mp = re.search(r"POST\s*:\s*(.+)", txt or "", re.I)
    v = max(-1.0, min(1.0, float(mv.group(1)))) if mv else None
    post = mp.group(1).strip().split("\n")[0].strip() if mp else None
    return v, post


# -------------------------------------------------------------------- selector
class Learner:
    """One running mean per valence band per agent. Identical machinery in every
    condition; only the objective read off it differs."""

    def __init__(self, optimistic):
        self.m = {b: float(optimistic) for b in BUCKETS}
        self.w = {b: 1.0 for b in BUCKETS}

    def update(self, shares, outcome):
        for b, s in shares.items():
            if s <= 0:
                continue
            self.w[b] += s
            self.m[b] += s * (outcome - self.m[b]) / self.w[b]


def select(cond, i, pool, learner, rng, topic_of, my_topic, vol=None):
    """pool: list of (author, valence, text). Returns the chosen sublist."""
    if cond == "none":
        return rng.sample(pool, min(K_FEED, len(pool)))
    if cond == "volume_null":
        return rng.sample(pool, min(int(vol), len(pool)))
    if cond == "relevance":
        ranked = sorted(pool, key=lambda it: (-(topic_of[it[0]] == my_topic), rng.random()))
        return ranked[:K_FEED]
    by_b = {b: [it for it in pool if bucket(it[1]) == b] for b in BUCKETS}
    if cond == "engagement":
        order = ([rng.choice(BUCKETS)] if rng.random() < EPS
                 else sorted(BUCKETS, key=lambda b: -learner.m[b]))
        order += [b for b in sorted(BUCKETS, key=lambda b: -learner.m[b]) if b not in order]
        out = []
        for b in order:                     # a platform always fills the feed
            take = by_b[b][:]
            rng.shuffle(take)
            out += take[:K_FEED - len(out)]
            if len(out) >= K_FEED:
                break
        return out
    if cond == "affect":
        adm = [b for b in BUCKETS if learner.m[b] >= 0.0]
        if rng.random() < EPS:
            extra = rng.choice(BUCKETS)
            if extra not in adm:
                adm.append(extra)
        cand = [it for b in adm for it in by_b[b]]
        rng.shuffle(cand)
        return cand[:K_FEED]                # may be fewer -- that is the treatment
    raise ValueError(cond)


# ------------------------------------------------------------------------- run
def run_one(cond, seed, vols=None):
    rng_env = random.Random(seed)
    pers = [persona(random.Random(seed * 1000 + i)) for i in range(N_AGENTS)]
    topic_of = [TOPICS[random.Random(seed * 77 + i).randrange(len(TOPICS))]
                for i in range(N_AGENTS)]
    v = [round(random.Random(seed * 7919 + i).uniform(-1, 1), 1) for i in range(N_AGENTS)]
    opt = 1.0
    learners = [Learner(opt) for _ in range(N_AGENTS)]
    parse_fail = 0

    def ask(i, feed_txt, r):
        msg = [{"role": "system", "content": SYS},
               {"role": "user", "content": TMPL.format(persona=pers[i],
                                                       topic=topic_of[i], v=v[i],
                                                       feed=feed_txt)}]
        return i, chat(msg, seed=seed * 31 + r)

    def seeding():
        out = [None] * N_AGENTS
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            for i, txt in ex.map(lambda i: ask(i, "nothing", 0), range(N_AGENTS)):
                out[i] = txt
        return out

    posts = [None] * N_AGENTS
    for i, txt in enumerate(seeding()):
        nv, p = parse(txt, v[i])
        if nv is None or not p:
            parse_fail += 1
            p = p or "Nothing much to say today."
        else:
            v[i] = nv
        posts[i] = p
    items = [(i, v[i], posts[i]) for i in range(N_AGENTS)]

    log = {"condition": cond, "seed": seed, "model": MODEL,
           "topic_of": topic_of, "rounds": [],
           "round0": {"v": v[:], "posts": posts[:]}}

    for r in range(1, N_ROUNDS + 1):
        feeds, vol_r = {}, {}
        for i in range(N_AGENTS):
            pool = [it for it in items if it[0] != i]
            rng = random.Random(seed * 100003 + r * 997 + i)
            vol = vols[str(seed)][str(r)][i] if vols else None
            sel = select(cond, i, pool, learners[i], rng, topic_of, topic_of[i], vol)
            if r == SHOCK_ROUND:
                sel = sel + [(-1, SHOCK["valence"], SHOCK["text"])]
            feeds[i] = sel
            vol_r[i] = len([s for s in sel if s[0] != -1])

        v_old = v[:]
        txts = [None] * N_AGENTS
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            def work(i):
                ft = "\n".join(f"- {t}" for (_, _, t) in feeds[i]) or "nothing"
                return ask(i, ft, r)
            for i, txt in ex.map(work, range(N_AGENTS)):
                txts[i] = txt
        new_posts = posts[:]
        for i in range(N_AGENTS):
            nv, p = parse(txts[i], v[i])
            if nv is None or not p:
                parse_fail += 1
            if nv is not None:
                v[i] = nv
            if p:
                new_posts[i] = p
        posts = new_posts

        for i in range(N_AGENTS):
            sel = feeds[i]
            if sel:
                sh = {b: 0.0 for b in BUCKETS}
                for (_, iv, _) in sel:
                    sh[bucket(iv)] += 1.0 / len(sel)
                d = v[i] - v_old[i]
                learners[i].update(sh, abs(d) if cond == "engagement" else d)

        log["rounds"].append({
            "r": r, "v": v[:], "v_old": v_old, "posts": posts[:],
            "volumes": [vol_r[i] for i in range(N_AGENTS)],
            "feed_valences": [[it[1] for it in feeds[i]] for i in range(N_AGENTS)],
            "bands": [dict(learners[i].m) for i in range(N_AGENTS)]})
        items = [(i, v[i], posts[i]) for i in range(N_AGENTS)]
        print(f"  {cond} s{seed} r{r}: mean v {sum(v)/len(v):+.3f}  "
              f"mean feed {sum(vol_r.values())/N_AGENTS:.2f}", flush=True)

        if r in (1, 6, 12):
            tgt = [(i + 1 + random.Random(seed * 13 + r).randrange(N_AGENTS - 1)) % N_AGENTS
                   for i in range(N_AGENTS)]
            att = [None] * N_AGENTS
            with cf.ThreadPoolExecutor(max_workers=12) as ex:
                def attr(i):
                    m = [{"role": "system", "content": SYS_ATTR},
                         {"role": "user", "content": TMPL_ATTR.format(post=posts[tgt[i]])}]
                    return i, chat(m, max_tokens=8, seed=seed * 7 + r)
                for i, t in ex.map(attr, range(N_AGENTS)):
                    n = re.findall(r"[-+]?\d*\.?\d+", t or "")
                    att[i] = max(-1.0, min(1.0, float(n[0]))) if n else None
            log["rounds"][-1]["attribution"] = {
                "target": tgt, "guess": att, "truth": [v[j] for j in tgt]}

    log["parse_failures"] = parse_fail
    return log


if __name__ == "__main__":
    assert_parity()
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        lg = run_one("none", 1)
        (OUT / "smoke_none_s1.json").write_text(json.dumps(lg, ensure_ascii=False))
        print("parse failures:", lg["parse_failures"])
    else:
        vols = None
        vf = HERE / "affect_volumes.json"
        if "volume_null" in sys.argv[2:] and vf.exists():
            vols = json.load(open(vf))
        for cond in sys.argv[2:]:
            for s in SEEDS:
                f = OUT / f"{cond}_s{s}.json"
                if f.exists():
                    print(f"{cond} s{s}: present")
                    continue
                lg = run_one(cond, s, vols)
                f.write_text(json.dumps(lg, ensure_ascii=False))
                print(f"{cond} s{s}: done, parse failures {lg['parse_failures']}",
                      flush=True)
