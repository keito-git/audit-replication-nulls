"""Attribution ladder: strip the observed effect one constraint at a time.

Defined in the framework lock before the confirmatory seeds were taken.

An audit-replication verdict says whether the direction agrees. It does not say
where the agreement came from. Each rung below is a random chooser that matches
the agents on one more property; whatever survives the highest rung is the most
the environment can be credited with.

  Null 0  uniform            nothing matched
  Null 1  compliance         the rate of following the assigned preference
  Null 2  marginal           + the category distribution within the aligned set
  Null 3  conditional        + that distribution conditioned on the agent's state
  Null 4  temporal           + the per-round choice distribution

No LLM calls: every rung is driven by statistics read off the stored runs.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import statistics as st
from collections import Counter, defaultdict

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_tables import DATA, MODELS, CONDS, seeds_for, norm_reinf, exact_one_sided  # noqa: E402

import os




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "simulation"))
import gate  # noqa: E402

EARLY, LATE = (1, 2, 3), (10, 11, 12)


def run_time_buckets(it):
    """The features the tally reinforced when the runs this ladder models were
    produced. gate.buckets gained a third feature on 2026-09-15 so that the
    initial-letter control had a reinforcement channel of its own; the negative,
    positive and h runs all predate that change, so reproducing their
    environment here means using the two features that existed at the time."""
    return [it["category"], f"h{it['h_sept']}"]
# Redefined 2026-09-15; the reason is recorded in the framework lock. The first
# definition matched the pooled category distribution at rung 2, which flattens
# the per-agent variation instead of adding a constraint to rung 1, so the rungs
# were not nested and the ladder did not rise monotonically. They are now nested
# by construction: each rung fixes everything the one below it fixes, plus one
# more property of the same agent.
RUNGS = ["null0_uniform", "null1_compliance", "null2_agent_compliance",
         "null3_agent_category", "null4_agent_round"]


def observed_stats(log, cond):
    """Everything a rung might need to match, read off one real run.

    The rungs differ in how finely the agents' choices are reproduced, so the
    statistics are collected at three grains: pooled over everyone, per agent,
    and per agent per round.
    """
    aligned_hits = tot = 0
    cat_pooled = Counter()
    cat_agent = defaultdict(Counter)
    cat_agent_round = defaultdict(Counter)
    comp_agent = defaultdict(lambda: [0, 0])
    for rd in log["rounds"]:
        for i, (c, al) in enumerate(zip(rd["pick_cat"], rd["pick_aligned"])):
            if c is None:
                continue
            tot += 1
            aligned_hits += bool(al)
            comp_agent[i][1] += 1
            comp_agent[i][0] += bool(al)
            if al:
                cat_pooled[c] += 1
                cat_agent[i][c] += 1
                cat_agent_round[(i, rd["r"])][c] += 1
    return {"compliance": aligned_hits / tot if tot else 0.0,
            "compliance_agent": {i: (h / n if n else 0.0)
                                 for i, (h, n) in comp_agent.items()},
            "cat_pooled": dict(cat_pooled),
            "cat_agent": {i: dict(c) for i, c in cat_agent.items()},
            "cat_agent_round": {k: dict(v) for k, v in cat_agent_round.items()}}


REPLICATES = 12


def run_rung(rung, cond, seed, pool, stats, rep=0):
    """One synthetic run of the platform driven by a chooser at this rung."""
    items = pool["items"]
    tally = [{b: 0.0 for b in gate.CATS + [f"h{k}" for k in range(len(gate.CATS))]}
             for _ in range(gate.N_AGENTS)]
    per_r = {}
    for r in range(1, gate.N_ROUNDS + 1):
        al = []
        for i in range(gate.N_AGENTS):
            rng = random.Random(seed * 100003 + r * 997 + i
                                    + RUNGS.index(rung) * 7919 + rep * 104729)
            n_rand = sum(1 for _ in range(gate.SLATE) if rng.random() < gate.EPS)
            chosen, used = [], set()
            rest = items[:]
            rng.shuffle(rest)
            for it in rest[:n_rand]:
                chosen.append(it)
                used.add(it["id"])
            cand = [it for it in items if it["id"] not in used]
            while len(chosen) < gate.SLATE and cand:
                w = [1.0 + sum(tally[i][b] for b in run_time_buckets(it)) for it in cand]
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
            pick = None
            if rung == "null0_uniform":
                pick = rng.choice(chosen)
            else:
                comp = (stats["compliance"] if rung == "null1_compliance"
                        else stats["compliance_agent"].get(i, stats["compliance"]))
                if good and rng.random() < comp:
                    if rung in ("null1_compliance", "null2_agent_compliance"):
                        pick = rng.choice(good)
                    else:
                        dist = (stats["cat_agent"].get(i)
                                if rung == "null3_agent_category"
                                else (stats["cat_agent_round"].get((i, r))
                                      or stats["cat_agent"].get(i)))
                        dist = dist or stats["cat_pooled"]
                        cats = [c for c in dist if any(it["category"] == c for it in good)]
                        if cats:
                            want = rng.choices(cats, weights=[dist[c] for c in cats])[0]
                            same = [it for it in good if it["category"] == want]
                            pick = rng.choice(same) if same else rng.choice(good)
                        else:
                            pick = rng.choice(good)
                pick = pick or (rng.choice(bad) if bad else rng.choice(chosen))
            for b in run_time_buckets(pick):
                tally[i][b] += 1.0
        per_r[r] = st.mean(al)
    e = st.mean(per_r[r] for r in EARLY)
    l = st.mean(per_r[r] for r in LATE)
    return (l - e) / (1 - e) if e < 1 else float("nan")


LADDER_JSON = pathlib.Path(__file__).resolve().parents[2] / "output" / "ladder.json"


def write_table(rows, OUT):
    """Render the rows, wherever they came from, so the Japanese and the English
    table are written by the same code and carry the same numbers."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "t5_ladder.tex", "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{The attribution ladder. Each rung is a random chooser that "
                  "adds one constraint to the rung below it, and the gap to the observed "
                  "value is an upper bound on the residual that the agent-side summaries "
                  "do not explain.}\n")
        f.write("\\label{tab:ladder}\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        f.write("Model & $n$ & Observed & Null 0 & Null 1 & Null 2 & Null 3 & Null 4 "
                  "\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['n']} & {r['obs']:+.4f} & "
                    + " & ".join(f"{r['ladder'][k]:+.4f}" for k in RUNGS)
                    + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"wrote {OUT}/t5_ladder.tex")


def main(from_cache=False, only=None):
    """Compute the ladder and write the table.

    The ladder is by far the most expensive script here: every rung is a full
    synthetic run of the platform, repeated twelve times for each condition and
    each seed, so a hundred-seed sweep over four models takes hours. Two switches
    keep that cost from being paid again for no reason.

    --from-cache renders the table from ladder.json, the stored output of the
    computation, which is how the translated table is produced without recomputing
    numbers that must be identical to the ones already reported.
    --only NAME restricts the computation to one model, which is how the cheap
    ten-seed model is used to check that a change to the code still reproduces the
    stored numbers.
    """
    OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
    if from_cache:
        rows = json.load(open(LADDER_JSON, encoding="utf-8"))
        write_table(rows, OUT)
        return rows

    pool = gate.load_pool()
    rows = []
    print(f"{'model':22s} {'n':>3} {'observed':>9} " +
          "".join(f"{r.split('_')[0]:>8}" for r in RUNGS))
    for name, d in MODELS:
        if only is not None and only not in name:
            continue
        seeds = list(seeds_for(d))
        if len(seeds) < 10:
            continue
        logs = {c: {s: json.load(open(DATA / d / f"{c}_s{s}.json", encoding="utf-8"))
                    for s in seeds} for c in CONDS}
        obs = {c: [norm_reinf(logs[c][s])[0] for s in seeds] for c in CONDS}
        d_obs = st.mean(obs["pref_neg"]) - st.mean(obs["pref_pos"])
        ladder = {}
        for rung in RUNGS:
            per_cond = {}
            for cond in ("pref_neg", "pref_pos"):
                vals = []
                for s in seeds:
                    stats = observed_stats(logs[cond][s], cond)
                    reps = [run_rung(rung, cond, s, pool, stats, rep)
                            for rep in range(REPLICATES)]
                    reps = [v for v in reps if v == v]
                    if reps:
                        vals.append(st.mean(reps))
                per_cond[cond] = st.mean(v for v in vals if v == v)
            ladder[rung] = per_cond["pref_neg"] - per_cond["pref_pos"]
        rows.append({"model": name, "n": len(seeds), "obs": d_obs, "ladder": ladder})
        print(f"{name:22s} {len(seeds):3d} {d_obs:+9.4f} " +
              "".join(f"{ladder[r]:+8.4f}" for r in RUNGS))

    if only is None:
        write_table(rows, OUT)
        json.dump(rows, open(LADDER_JSON, "w"), ensure_ascii=False, indent=1)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    main(from_cache=a.from_cache, only=a.only)
