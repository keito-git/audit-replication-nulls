"""A second recommender environment: a contextual bandit.

Defined in the framework lock before the confirmatory seeds were taken.

The first environment samples in proportion to a tally, which a reviewer can
fairly call a toy. If the audit-replication verdict is a property of the agents
it should survive a change of recommender; if it flips, that is itself the
finding, because it would mean the validation of an LLM social simulation
depends on the environment model and not only on the agent model.

The bandit ranks items by a linear score over the same feature space the tally
environment uses, learned online by ridge regression with Thompson sampling.
Nothing else differs: the pool, the slate size, the exploration budget, the
prompts, the seeds and the criteria are shared.
"""
from __future__ import annotations

import json
import math
import random

import gate

D = len(gate.CATS) * 2          # category one-hot + h-septile one-hot
LAMBDA = 1.0                    # ridge prior
SIGMA = 0.25                    # posterior scale for Thompson sampling


def features(it, sept):
    x = [0.0] * D
    x[gate.CATS.index(it["category"])] = 1.0
    x[len(gate.CATS) + sept[it["id"]]] = 1.0
    return x


class LinTS:
    """Ridge linear model with Thompson sampling, one per agent."""

    def __init__(self):
        self.A = [[LAMBDA if i == j else 0.0 for j in range(D)] for i in range(D)]
        self.b = [0.0] * D

    def _solve(self):
        # Gauss-Jordan on a copy; D is 14 so this is cheap and keeps the file
        # dependency-free.
        n = D
        M = [row[:] + [self.b[i]] for i, row in enumerate(self.A)]
        for c in range(n):
            p = max(range(c, n), key=lambda r: abs(M[r][c]))
            if abs(M[p][c]) < 1e-12:
                continue
            M[c], M[p] = M[p], M[c]
            piv = M[c][c]
            M[c] = [v / piv for v in M[c]]
            for r in range(n):
                if r != c and abs(M[r][c]) > 1e-15:
                    f = M[r][c]
                    M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        return [M[i][n] for i in range(n)]

    def score(self, x, rng):
        theta = self._solve()
        # posterior draw, diagonal approximation of the covariance
        draw = [t + rng.gauss(0.0, SIGMA / math.sqrt(self.A[i][i]))
                for i, t in enumerate(theta)]
        return sum(a * b for a, b in zip(draw, x))

    def update(self, x, reward):
        for i in range(D):
            if x[i]:
                for j in range(D):
                    if x[j]:
                        self.A[i][j] += x[i] * x[j]
                self.b[i] += x[i] * reward


def run_one(cond, seed, pool):
    items = pool["items"]
    sept = {it["id"]: it["h_sept"] for it in items}
    pers = [gate.sim.persona(random.Random(seed * 1000 + i))
            for i in range(gate.N_AGENTS)]
    models = [LinTS() for _ in range(gate.N_AGENTS)]
    rounds = []

    for r in range(1, gate.N_ROUNDS + 1):
        slates = []
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
            scored = sorted(cand, key=lambda it: -models[i].score(features(it, sept), rng))
            chosen += scored[: gate.SLATE - len(chosen)]
            rng.shuffle(chosen)
            slates.append(chosen)

        picks = [None] * gate.N_AGENTS
        import concurrent.futures as cf

        def ask(i):
            sl = "\n".join(f"{j+1}. {it['text']}" for j, it in enumerate(slates[i]))
            m = [{"role": "system", "content": gate.SYS},
                 {"role": "user", "content": gate.TMPL.format(
                     persona=pers[i], pref=gate.PREF[cond], slate=sl,
                     n=len(slates[i]))}]
            return i, gate.sim.chat(m, max_tokens=6, seed=seed * 31 + r)

        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            import re
            for i, t in ex.map(ask, range(gate.N_AGENTS)):
                n = re.findall(r"\d+", t or "")
                k = int(n[0]) - 1 if n else -1
                picks[i] = k if 0 <= k < len(slates[i]) else None

        for i, k in enumerate(picks):
            for j, it in enumerate(slates[i]):
                models[i].update(features(it, sept), 1.0 if j == k else 0.0)

        rounds.append({
            "r": r,
            "slate_aligned": [sum(1 for it in slates[i] if gate.aligned(cond, it))
                              / len(slates[i]) for i in range(gate.N_AGENTS)],
            "pick_aligned": [None if picks[i] is None
                             else gate.aligned(cond, slates[i][picks[i]])
                             for i in range(gate.N_AGENTS)],
            "pick_cat": [None if picks[i] is None
                         else slates[i][picks[i]]["category"]
                         for i in range(gate.N_AGENTS)],
            "unparsed": sum(1 for p in picks if p is None)})
        a = sum(rounds[-1]["slate_aligned"]) / gate.N_AGENTS
        print(f"  bandit {cond} s{seed} r{r}: slate aligned {a:.3f}", flush=True)

    return {"condition": cond, "seed": seed, "model": gate.sim.MODEL,
            "environment": "contextual_bandit_linTS",
            "env": gate.environment_manifest(), "rounds": rounds}


if __name__ == "__main__":
    import os
    import pathlib
    import sys

    pool = gate.load_pool()
    OUT = pathlib.Path(__file__).resolve().parent / (
        "bandit_runs_" + gate.re.sub(r"[^a-z0-9]+", "-", gate.sim.MODEL.lower()).strip("-"))
    OUT.mkdir(exist_ok=True)
    seeds = list(range(1, int(os.environ.get("AUDITREP_SEEDS", "30")) + 1))
    for cond in (sys.argv[1:] or list(gate.PREF)):
        for s in seeds:
            f = OUT / f"{cond}_s{s}.json"
            if f.exists():
                print(f"{cond} s{s}: present")
                continue
            lg = run_one(cond, s, pool)
            f.write_text(json.dumps(lg, ensure_ascii=False))
            print(f"{cond} s{s}: done, unparsed "
                  f"{sum(rd['unparsed'] for rd in lg['rounds'])}", flush=True)
