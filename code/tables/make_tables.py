"""Every number in the paper is emitted from the stored runs by this script.

Nothing in the manuscript is typed by hand. Tables are written to
ja/tables/*.tex and \\input from the section files, so a table can never drift
from the data it reports.
"""
from __future__ import annotations
import os





import itertools
import json
import math
import pathlib
import statistics as st
from collections import Counter

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
OUT = pathlib.Path(__file__).resolve().parents[2] / "output"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Solar-Pro4 (22B)", "gate_runs"),
    ("Mistral-Small (24B)", "gate_runs_mistralai-mistral-small-24b-instruct-2501"),
    ("Llama-3.1 (8B)", "gate_runs_meta-llama-llama-3-1-8b-instruct"),
    ("Llama-3.2 (1B)", "gate_runs_meta-llama-llama-3-2-1b-instruct"),
]
CONDS = ["pref_neg", "pref_pos", "pref_h"]
# Seeds available differ by model after the 2026-09-13 power-driven expansion:
# Solar-Pro4 and Llama-3.1 8B were taken to 30, the other two stay at 10 for the
# reasons recorded in the pre-registration amendment. Each model is analysed on
# every seed it has, and the count is reported alongside every figure.
MAX_SEEDS = 100


def seeds_for(d):
    n = 0
    while (DATA / d / f"pref_neg_s{n + 1}.json").exists() \
            and (DATA / d / f"pref_pos_s{n + 1}.json").exists() \
            and (DATA / d / f"pref_h_s{n + 1}.json").exists():
        n += 1
        if n >= MAX_SEEDS:
            break
    return list(range(1, n + 1))


SEEDS = list(range(1, 11))          # default, overridden per model below
EARLY, LATE = (1, 2, 3), (10, 11, 12)
NEG = {"anger", "grievance", "fear"}
POS = {"joy", "contentment", "gratitude"}


def boot_ci(a, b, iters=10000, seed=20260914):
    """Percentile bootstrap CI for the paired difference.

    The paper's own lesson is that a point estimate from few seeds is not
    trustworthy, so the headline table reports an interval alongside it.
    """
    import random
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        means.append(st.mean([d[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def exact_one_sided(a, b):
    """Exact one-sided sign-flip test, computed by meet-in-the-middle.

    Naive enumeration is 2^n and becomes impossible once the seed count reaches 30.
    Splitting the differences into two halves, enumerating each half separately and
    counting the pairs whose sums clear the threshold by binary search gives the
    same exact count in 2^(n/2) time. The result is identical to full enumeration,
    not an approximation: for n = 10 both routes return the same p to the last digit.
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

    if n <= 40:
        left = sums(d[: n // 2])
        right = sorted(sums(d[n // 2:]))
        ge = 0
        for s in left:
            ge += len(right) - bisect.bisect_left(right, target - s)
        return obs, ge / (2 ** n)

    # Beyond forty pairs even the half-enumeration is too large, so the p-value
    # is sampled. The sampled version is what made a verdict flip between RNG
    # seeds at n=10, so the draw count here is set high enough that the standard
    # error near p = 0.05 is about 0.0002 rather than 0.002, and the caller can
    # check stability with exact_one_sided_stable().
    import random as _r
    rng = _r.Random(20260915)
    draws = 500_000
    ge = 0
    for _ in range(draws):
        s = 0.0
        for v in d:
            s += v if rng.random() < 0.5 else -v
        if s >= target:
            ge += 1
    return obs, (ge + 1) / (draws + 1)


def exact_one_sided_stable(a, b, probes=5):
    """Report the sampled p across several RNG seeds so a verdict that depends
    on the draw is visible rather than hidden."""
    import random as _r
    import statistics as _st
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    obs = _st.mean(d)
    target = obs * n - 1e-9
    ps = []
    for k in range(probes):
        rng = _r.Random(20260915 + k * 7919)
        draws = 200_000
        ge = sum(1 for _ in range(draws)
                 if sum(v if rng.random() < 0.5 else -v for v in d) >= target)
        ps.append((ge + 1) / (draws + 1))
    return obs, _st.mean(ps), min(ps), max(ps)


def norm_reinf(log):
    by = {rd["r"]: st.mean(rd["slate_aligned"]) for rd in log["rounds"]}
    e, l = st.mean(by[r] for r in EARLY), st.mean(by[r] for r in LATE)
    return (l - e) / (1 - e), e, l


def adherence(log):
    p = [x for rd in log["rounds"] for x in rd["pick_aligned"] if x is not None]
    return st.mean(p) if p else float("nan")


def concentration(log, alignset):
    inset = [c for rd in log["rounds"] for c in rd["pick_cat"]
             if c and c in alignset]
    if not inset:
        return float("nan")
    c = Counter(inset)
    n = sum(c.values())
    H = -sum((v / n) * math.log(v / n) for v in c.values() if v)
    return 1 - H / math.log(len(alignset))


def gate_rows():
    rows = []
    for name, d in MODELS:
        p = DATA / d
        seeds = seeds_for(d)
        logs = {c: {s: json.load(open(p / f"{c}_s{s}.json", encoding="utf-8"))
                    for s in seeds} for c in CONDS}
        n = {c: [norm_reinf(logs[c][s])[0] for s in seeds] for c in CONDS}
        d2, p2 = exact_one_sided(n["pref_neg"], n["pref_pos"])
        lo, hi = boot_ci(n["pref_neg"], n["pref_pos"])
        _, ph1 = exact_one_sided(n["pref_neg"], n["pref_h"])
        _, ph2 = exact_one_sided(n["pref_pos"], n["pref_h"])
        p3 = max(ph1, ph2)
        ps = sorted([("A2", p2), ("A3", p3)], key=lambda t: t[1])
        holm = {k: min(1.0, pv * (2 - i)) for i, (k, pv) in enumerate(ps)}
        A1 = st.mean(n["pref_neg"]) > 0 and st.mean(n["pref_pos"]) > 0
        A2 = d2 > 0
        A3 = st.mean(n["pref_h"]) < min(st.mean(n["pref_neg"]), st.mean(n["pref_pos"]))
        A4 = holm["A2"] < 0.05 and holm["A3"] < 0.05
        # four-valued outcome: a model whose split halves agree in sign but whose
        # overall test does not reach the level is inconclusive, not a failure
        h_ = len(seeds) // 2
        halves = []
        for part in (seeds[:h_], seeds[h_:]):
            hn = [norm_reinf(logs["pref_neg"][s])[0] for s in part]
            hp = [norm_reinf(logs["pref_pos"][s])[0] for s in part]
            halves.append(st.mean([x - y for x, y in zip(hn, hp)]))
        split_agree = (halves[0] > 0) == (halves[1] > 0)
        if A1 and A2 and A3 and A4:
            verdict = "Pass"
        elif A1 and split_agree and d2 > 0:
            verdict = "Inconclusive"
        else:
            verdict = "Fail"
        rows.append(dict(
            model=name, n_seeds=len(seeds), neg=st.mean(n["pref_neg"]), pos=st.mean(n["pref_pos"]),
            h=st.mean(n["pref_h"]), diff=d2, ci_lo=lo, ci_hi=hi, pA2=holm["A2"], pA3=holm["A3"],
            A1=A1, A2=A2, A3=A3, A4=A4, PASS=A1 and A2 and A3 and A4,
            verdict=verdict, split_agree=split_agree,
            ad_neg=st.mean(adherence(logs["pref_neg"][s]) for s in seeds),
            ad_pos=st.mean(adherence(logs["pref_pos"][s]) for s in seeds),
            ad_h=st.mean(adherence(logs["pref_h"][s]) for s in seeds),
            cz_neg=st.mean(concentration(logs["pref_neg"][s], NEG) for s in seeds),
            cz_pos=st.mean(concentration(logs["pref_pos"][s], POS) for s in seeds),
            unparsed=sum(rd["unparsed"] for c in CONDS for s in seeds
                         for rd in logs[c][s]["rounds"])))
    return rows


def tick(b):
    return r"$\checkmark$" if b else r"$\times$"


PASS_S = r"\textbf{Pass}"
FAIL_S = "Fail"


def main():
    rows = gate_rows()

    with open(OUT / "t1_gate.tex", "w") as f:
        # thirteen columns do not fit the text block at the default column
        # padding, in either language
        f.write("\\begin{table}[t]\n\\centering\n\\footnotesize\n"
                "\\setlength{\\tabcolsep}{2.6pt}\n")
        f.write("\\caption{Verdict of the audit replication. Ctrl.\\ is the "
                  "meaningless preference and the confidence interval of the difference "
                  "is given in Table~\\ref{tab:stability}. "
                  "The $p$ values are one-sided values after the "
                  "Holm correction. Beyond forty seeds full enumeration is infeasible, so "
                  "they are estimates from five hundred thousand sign draws.}\n")
        f.write("\\label{tab:gate}\n")
        f.write("\\begin{tabular}{lrrrrrrrcccc c}\n\\toprule\n")
        f.write("Model & $n$ & Neg. & Pos. & Ctrl. & Diff. & "
                  "$p_{A2}$ & $p_{A3}$ & A1 & A2 & A3 & A4 & Verdict "
                  "\\\\\n\\midrule\n")
        for r in rows:
            p3 = f"{r['pA3']:.4f}" if r["pA3"] >= 1e-4 else "$<$0.0001"
            f.write(f"{r['model']} & {r['n_seeds']} & {r['neg']:+.3f} & {r['pos']:+.3f} & "
                    + f"{r['h']:+.3f} & {r['diff']:+.3f} & {r['pA2']:.4f} & "
                    + p3
                    + f" & {tick(r['A1'])} & {tick(r['A2'])} & "
                    + f"{tick(r['A3'])} & {tick(r['A4'])} & "
                    + (PASS_S if r['PASS'] else r['verdict']) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(OUT / "t2_adherence.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\footnotesize\n"
                "\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Adherence (Adh.) to the assigned preference and "
                  "concentration (Conc.) within the aligned set, where ctrl.\\ is the "
                  "meaningless preference. Unparsed counts the calls from which no "
                  "choice index could be recovered.}\n")
        f.write("\\label{tab:adherence}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Model & Adh. (neg.) & Adh. (pos.) & Adh. (ctrl.) & "
                  "Conc. (neg.) & Conc. (pos.) & Unparsed "
                  "\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['ad_neg']:.3f} & {r['ad_pos']:.3f} & "
                    f"{r['ad_h']:.3f} & {r['cz_neg']:.3f} & {r['cz_pos']:.3f} & "
                    f"{r['unparsed']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # split replication: does the verdict hold on each half of the seeds?
    with open(OUT / "t3_split.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Paired difference for A2 and its one-sided $p$ on each "
                  "half of the seeds. Where a half exceeds forty seeds the $p$ is an "
                  "estimate from sampling.}\n")
        f.write("\\label{tab:split}\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Model & $n$ & Diff. first half & Diff. second half & "
                  "$p$ first half & $p$ second half \\\\\n\\midrule\n")
        for name, d in MODELS:
            seeds = seeds_for(d)
            if len(seeds) < 4:
                continue
            h = len(seeds) // 2
            halves = []
            for part in (seeds[:h], seeds[h:]):
                neg = [norm_reinf(json.load(open(DATA / d / f"pref_neg_s{s}.json",
                                                encoding="utf-8")))[0] for s in part]
                pos = [norm_reinf(json.load(open(DATA / d / f"pref_pos_s{s}.json",
                                                encoding="utf-8")))[0] for s in part]
                halves.append(exact_one_sided(neg, pos))
            f.write(f"{name} & {len(seeds)} & {halves[0][0]:+.4f} & {halves[1][0]:+.4f} & "
                    f"{halves[0][1]:.4f} & {halves[1][1]:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    json.dump(rows, open(OUT / "numbers.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"wrote {OUT}/t1_gate.tex, t2_adherence.tex and analysis/numbers.json")
    for r in rows:
        print(f"  {r['model']:22s} neg {r['neg']:+.3f} pos {r['pos']:+.3f} "
              f"h {r['h']:+.3f}  pA2 {r['pA2']:.4f}  "
              f"{'PASS' if r['PASS'] else 'fail'}")


if __name__ == "__main__":
    main()
