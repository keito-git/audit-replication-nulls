"""Result figures, generated from the stored runs. Times New Roman throughout,
per the lab standard."""
from __future__ import annotations

import json
import os
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
    "font.size": 9,
})

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
FIG = HERE.parents[1] / "output"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Solar-Pro4 (22B)", "gate_runs"),
    ("Mistral-Small (24B)", "gate_runs_mistralai-mistral-small-24b-instruct-2501"),
    ("Llama-3.1 (8B)", "gate_runs_meta-llama-llama-3-1-8b-instruct"),
    ("Llama-3.2 (1B)", "gate_runs_meta-llama-llama-3-2-1b-instruct"),
]
CONDS = [("pref_neg", "negative"), ("pref_pos", "positive"),
         ("pref_h", "meaningless")]
STYLE = {"pref_neg": ("k", "-", "o"), "pref_pos": ("0.45", "--", "s"),
         "pref_h": ("0.7", ":", "^")}
def seeds_for(d):
    n = 0
    while all((DATA / d / f"{c}_s{n + 1}.json").exists()
              for c in ("pref_neg", "pref_pos", "pref_h")) and n < 100:
        n += 1
    return range(1, n + 1)


def curves(d, cond):
    per_r = {r: [] for r in range(1, 13)}
    for s in seeds_for(d):
        log = json.load(open(DATA / d / f"{cond}_s{s}.json", encoding="utf-8"))
        for rd in log["rounds"]:
            per_r[rd["r"]].append(st.mean(rd["slate_aligned"]))
    rs = sorted(per_r)
    return rs, [st.mean(per_r[r]) for r in rs], [st.pstdev(per_r[r]) for r in rs]


fig, axes = plt.subplots(1, 4, figsize=(11, 2.7), sharey=True)
for ax, (name, d) in zip(axes, MODELS):
    for cond, lab in CONDS:
        c, ls, mk = STYLE[cond]
        rs, m, sd = curves(d, cond)
        ax.plot(rs, m, color=c, linestyle=ls, marker=mk, markersize=3,
                linewidth=1.2, label=lab)
        ax.fill_between(rs, [a - b for a, b in zip(m, sd)],
                        [a + b for a, b in zip(m, sd)], color=c, alpha=0.12,
                        linewidth=0)
    ax.axhline(30 / 70, color="r", linewidth=0.7, linestyle="-.", alpha=0.6)
    ax.set_title(f"{name}  $n$={len(seeds_for(d))}", fontsize=9)
    ax.set_xlabel("round")
    ax.set_xlim(1, 12)
    ax.set_xticks([1, 4, 8, 12])
    ax.grid(alpha=0.25, linewidth=0.5)
axes[0].set_ylabel("share of the slate aligned\nwith the assigned preference")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig2_reinforcement.pdf", bbox_inches="tight")
print("wrote fig2_reinforcement.pdf")

# observed asymmetry against the distribution-matched baseline: the paper's core measure
rows = json.load(open(HERE.parents[1] / "output" / "numbers.json"))
# The baseline is the rung of the attribution ladder that comes closest to the
# observation, read from ladder.json so the figure cannot drift from the table.
_L = {r["model"]: r for r in json.load(open(HERE.parents[1] / "output" / "ladder.json"))}
NULL = {m: (max(v["ladder"].values()) if v["n"] >= 30 else None)
        for m, v in _L.items()}
fig, ax = plt.subplots(figsize=(5.0, 3.1))
names = [r["model"] for r in rows if NULL.get(r["model"]) is not None]
obs = [r["diff"] for r in rows if NULL.get(r["model"]) is not None]
nul = [NULL[n] for n in names]
y = list(range(len(names)))
ax.barh([v + 0.18 for v in y], obs, height=0.34, color="0.25",
        edgecolor="k", linewidth=0.6, label="observed")
ax.barh([v - 0.18 for v in y], nul, height=0.34, color="0.75",
        edgecolor="k", linewidth=0.6, label="distribution-matched null")
ax.axvline(0, color="0.3", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("asymmetry between the negative and positive preference")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.grid(axis="x", alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig3_adherence.pdf", bbox_inches="tight")
print("wrote fig3_adherence.pdf (observed vs matched null)")
