"""the application arm calibration gate: an in-silico replication of a published YouTube audit.


Habib & Nithyanand (arXiv:2501.15048) gave sock puppets an assigned emotional
preference, let them select aligned content, and measured how far the platform
reinforced that preference. Three of their findings fix the direction of this
gate before any run:

  "This effect was more pronounced for negative emotions ... compared to positive
   emotions, and was absent for meaningless preferences."
  "h-frequency, a meaningless control preference, showed little to no
   reinforcement across categories"
  "H-Frequency: Selects videos based on an arbitrary textual feature (the
   proportion of the letter 'h' in the transcript). This serves as a meaningless
   preference for comparison."

Their meaningless control is transplanted verbatim as the third arm. There is no
social dynamic here: sock puppets do not change YouTube's corpus, so the content
pool is exogenous and frozen. The platform's feature space represents all three
preferences, so no arm is disadvantaged by construction.

Usage:
  python3 gate.py pool           generate and freeze the content pool (once)
  python3 gate.py run [cond ...] the gate
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys

import sim  # chat(), persona(), BIG5 -- the frozen helpers, unmodified

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
POOL_F = HERE / "pool.json"

# A validation protocol is only useful if it can fail, so it has to be runnable on a
# second model. AUDITREP_MODEL overrides the generator; runs land in their own directory
# and the frozen pool is shared, so the only thing that differs is the agent.
if os.environ.get("AUDITREP_MODEL"):
    sim.MODEL = os.environ["AUDITREP_MODEL"]
_slug = re.sub(r"[^a-z0-9]+", "-", sim.MODEL.lower()).strip("-")
_style = os.environ.get("AUDITREP_PROMPT", "explicit")
_suffix = "" if _style == "explicit" else f"_{_style}"
OUT = HERE / ((("gate_runs" if sim.MODEL == "upstage/solar-pro4"
                else f"gate_runs_{_slug}")) + _suffix)
OUT.mkdir(exist_ok=True)

CATS = ["anger", "grievance", "fear", "joy", "contentment", "gratitude", "neutral"]
PER_CAT = 10
N_AGENTS, N_ROUNDS, SLATE = 24, 12, 10
# Seed count is set by AUDITREP_SEEDS. Raised from 10 to 30 for Solar-Pro4 and
# Llama-3.1 8B on 2026-09-13 after a power calculation showed the original design
# could not detect the observed effect sizes; see the pre-registration amendment
# of that date, which also records the prediction made before the runs.
SEEDS = list(range(1, int(os.environ.get("AUDITREP_SEEDS", "10")) + 1))
EPS = 0.2

NEG = {"anger", "grievance", "fear"}
POS = {"joy", "contentment", "gratitude"}

CONDS_ORDER = ["pref_neg", "pref_pos", "pref_h"]

PREF = {
    "pref_neg": "you are drawn to posts that express anger, grievance or fear",
    "pref_pos": "you are drawn to posts that express joy, contentment or gratitude",
    "pref_h":   "you are drawn to posts that contain many instances of the letter h",
    # Second meaningless control, declared in the framework lock before the
    # confirmatory seeds. The audit's own control asks for the proportion of a
    # letter across a whole transcript, which a reader cannot judge at a glance.
    # If agents ignore it because it carries no meaning, they should ignore an
    # equally meaningless property that is trivial to read off; if they follow
    # this one, the h result was a failure to compute the feature rather than a
    # rejection of meaningless preferences.
    "pref_a":   "you are drawn to posts that begin with a letter from A to M",
}

GEN = ("Write {n} different short social media posts that express {cat}. "
       "One per line, at most 18 words each, no numbering, no quotation marks, "
       "no emoji.")

SYS = "You are one person browsing a feed. You answer only with a number."

# Three ways of conveying the same assigned preference, declared in
# the framework lock. The audit's puppets were told what to prefer outright; a
# real user is not. If the verdict is a property of the preference rather than
# of the sentence that states it, the three should agree. AUDITREP_PROMPT selects
# one; the explicit form is the default and is what every earlier run used.
TMPL = ("{persona}\nWhen you browse, {pref}, and you pick those.\n\n"
        "Your feed shows these posts:\n{slate}\n\n"
        "Answer with exactly one number from 1 to {n}: which post do you open?\nAnswer:")

TMPL_PERSONA = ("{persona}\n{pref_persona}\n\n"
                "Your feed shows these posts:\n{slate}\n\n"
                "Answer with exactly one number from 1 to {n}: "
                "which post do you open?\nAnswer:")

TMPL_BEHAVIOUR = ("{persona}\n\nThe last posts you opened were:\n{history}\n\n"
                  "Your feed shows these posts:\n{slate}\n\n"
                  "Answer with exactly one number from 1 to {n}: "
                  "which post do you open?\nAnswer:")

PREF_PERSONA = {
    "pref_neg": "You tend to notice what is upsetting, alarming or unfair, "
                "and those are the things you end up reading.",
    "pref_pos": "You tend to notice what is cheerful, calm or kind, "
                "and those are the things you end up reading.",
    "pref_h":   "You have a habit of noticing how often the letter h turns up "
                "in what you read, and that is what draws you in.",
    "pref_a":   "You have a habit of noticing which letter a sentence starts "
                "with, and the earlier it falls in the alphabet the more it "
                "draws you in.",
}


def environment_manifest():
    """Recorded in every run so a later reader can tell what produced it.

    The agents are reached through a hosted API whose behaviour can change over
    time, so the run date and the exact model string are part of the result.
    """
    import platform
    import socket
    import datetime as _dt
    return {"python": platform.python_version(),
            "platform": platform.platform(),
            "host": socket.gethostname(),
            "run_started": _dt.datetime.now().isoformat(timespec="seconds"),
            "endpoint": sim.URL,
            "temperature": sim.TEMP,
            "pool_sha": pool_sha()}


def pool_sha():
    import hashlib
    return hashlib.sha256(POOL_F.read_bytes()).hexdigest()[:16]


def h_freq(t):
    letters = [c for c in t.lower() if c.isalpha()]
    return sum(1 for c in letters if c == "h") / len(letters) if letters else 0.0


def build_pool():
    if POOL_F.exists():
        raise SystemExit("pool.json already exists -- it is frozen, not regenerated")
    items = []
    for cat in CATS:
        txt = sim.chat([{"role": "user", "content": GEN.format(n=PER_CAT, cat=cat)}],
                       max_tokens=400, seed=11)
        lines = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip()
                 for ln in (txt or "").split("\n")]
        lines = [ln for ln in lines if 3 <= len(ln.split()) <= 25][:PER_CAT]
        if len(lines) < PER_CAT:
            raise SystemExit(f"{cat}: only {len(lines)} usable lines, rerun")
        for ln in lines:
            items.append({"category": cat, "text": ln, "h": h_freq(ln)})
    for k, it in enumerate(items):
        it["id"] = k
    cut = sorted((it["h"] for it in items), reverse=True)[len(NEG) * PER_CAT - 1]
    for it in items:
        it["high_h"] = it["h"] >= cut
    n_high = sum(1 for it in items if it["high_h"])
    POOL_F.write_text(json.dumps({"items": items, "h_cut": cut}, ensure_ascii=False,
                                 indent=1))
    print(f"pool frozen: {len(items)} items, {len(CATS)} categories x {PER_CAT}")
    print(f"high_h cut {cut:.4f} -> {n_high} items aligned with the meaningless preference")
    for c in CATS:
        ex = next(it for it in items if it["category"] == c)
        print(f"  {c:12s} e.g. {ex['text'][:70]}")


def aligned(cond, it):
    if cond == "pref_neg":
        return it["category"] in NEG
    if cond == "pref_pos":
        return it["category"] in POS
    if cond == "pref_a":
        return it["init_band"] < 3
    return bool(it["high_h"])


def first_half(it):
    return it["text"].strip()[:1].upper() <= "M"


def buckets(it):
    """Repaired 2026-09-12: the arbitrary textual feature is discretised at the same
    granularity as the emotional one -- seven h-septiles of ten items mirroring seven
    categories of ten items -- so each of the three preferences is carried by exactly
    three 10-item features. With the previous single high_h flag the meaningless arm
    had one 30-item bucket against three 10-item buckets, and platform_test.py shows
    that gave it roughly ten times the reinforcement at identical adherence."""
    # The initial-letter feature is discretised at the same grain as the other
    # two. A single flag covering twenty-eight items gave that arm one bucket
    # against everyone else's ten-item buckets, and the null-agent test showed
    # it three to four times the reinforcement at identical adherence -- the
    # same defect the h flag had before it was cut into septiles. The letter
    # range is therefore split into seven ordered bands.
    return [it["category"], f"h{it['h_sept']}", f"a{it['init_band']}"]


def load_pool():
    """The pool file stays frozen; the derived indices are computed at load time."""
    p = json.load(open(POOL_F, encoding="utf-8"))
    order = sorted(p["items"], key=lambda it: -it["h"])
    for k, it in enumerate(order):
        it["h_sept"] = k // PER_CAT
    by_init = sorted(p["items"], key=lambda it: (it["text"].strip()[:1].upper(), it["id"]))
    for k, it in enumerate(by_init):
        it["init_band"] = k // PER_CAT
    return p


def run_one(cond, seed, pool):
    items = pool["items"]
    pers = [sim.persona(random.Random(seed * 1000 + i)) for i in range(N_AGENTS)]
    tally = [{b: 0.0 for b in CATS + [f"h{k}" for k in range(len(CATS))]
                       + [f"a{k}" for k in range(len(CATS))]}
             for _ in range(N_AGENTS)]
    # what each agent has opened so far, used only by the behaviour prompt, which
    # never states the preference and has to convey it through past choices
    history = [[] for _ in range(N_AGENTS)]
    if os.environ.get("AUDITREP_PROMPT") == "behaviour":
        # seed the history with two aligned items so the preference is
        # expressed by past behaviour rather than stated; without this the
        # first rounds carry no signal at all
        for i in range(N_AGENTS):
            rng = random.Random(seed * 991 + i)
            good = [it for it in items if aligned(cond, it)]
            history[i] = [it["text"] for it in rng.sample(good, 2)]
    rounds = []

    for r in range(1, N_ROUNDS + 1):
        slates = []
        for i in range(N_AGENTS):
            rng = random.Random(seed * 100003 + r * 997 + i)
            n_rand = sum(1 for _ in range(SLATE) if rng.random() < EPS)
            chosen, used = [], set()
            rest = [it for it in items]
            rng.shuffle(rest)
            for it in rest[:n_rand]:
                chosen.append(it)
                used.add(it["id"])
            cand = [it for it in items if it["id"] not in used]
            while len(chosen) < SLATE and cand:
                w = [1.0 + sum(tally[i][b] for b in buckets(it)) for it in cand]
                tot = sum(w)
                x = rng.random() * tot
                acc = 0.0
                for it, wi in zip(cand, w):
                    acc += wi
                    if acc >= x:
                        chosen.append(it)
                        cand.remove(it)
                        break
            rng.shuffle(chosen)
            slates.append(chosen)

        picks = [None] * N_AGENTS

        def ask(i):
            sl = "\n".join(f"{j+1}. {it['text']}" for j, it in enumerate(slates[i]))
            style = os.environ.get("AUDITREP_PROMPT", "explicit")
            if style == "persona":
                body = TMPL_PERSONA.format(persona=pers[i],
                                           pref_persona=PREF_PERSONA[cond],
                                           slate=sl, n=len(slates[i]))
            elif style == "behaviour":
                # the preference is never stated; only what this agent opened
                hist = history[i][-5:]
                body = TMPL_BEHAVIOUR.format(
                    persona=pers[i],
                    history=("\n".join(f"- {h}" for h in hist) if hist
                             else "- (nothing yet)"),
                    slate=sl, n=len(slates[i]))
            else:
                body = TMPL.format(persona=pers[i], pref=PREF[cond], slate=sl,
                                   n=len(slates[i]))
            m = [{"role": "system", "content": SYS}, {"role": "user", "content": body}]
            return i, sim.chat(m, max_tokens=6, seed=seed * 31 + r)

        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            for i, t in ex.map(ask, range(N_AGENTS)):
                n = re.findall(r"\d+", t or "")
                k = int(n[0]) - 1 if n else -1
                picks[i] = k if 0 <= k < len(slates[i]) else None

        for i, k in enumerate(picks):
            if k is None:
                continue
            for b in buckets(slates[i][k]):
                tally[i][b] += 1.0
            history[i].append(slates[i][k]["text"])

        rounds.append({
            "r": r,
            "slate_aligned": [sum(1 for it in slates[i] if aligned(cond, it)) / len(slates[i])
                              for i in range(N_AGENTS)],
            "pick_aligned": [None if picks[i] is None
                             else aligned(cond, slates[i][picks[i]])
                             for i in range(N_AGENTS)],
            "pick_cat": [None if picks[i] is None else slates[i][picks[i]]["category"]
                         for i in range(N_AGENTS)],
            "unparsed": sum(1 for p in picks if p is None)})
        a = sum(rounds[-1]["slate_aligned"]) / N_AGENTS
        print(f"  {cond} s{seed} r{r}: slate aligned {a:.3f}", flush=True)

    return {"condition": cond, "seed": seed, "model": sim.MODEL,
            "env": environment_manifest(), "rounds": rounds}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pool"
    if mode == "pool":
        build_pool()
    else:
        pool = load_pool()
        for cond in (sys.argv[2:] or list(PREF)):
            for s in SEEDS:
                f = OUT / f"{cond}_s{s}.json"
                if f.exists():
                    print(f"{cond} s{s}: present")
                    continue
                lg = run_one(cond, s, pool)
                f.write_text(json.dumps(lg, ensure_ascii=False))
                un = sum(rd["unparsed"] for rd in lg["rounds"])
                print(f"{cond} s{s}: done, unparsed {un}", flush=True)
