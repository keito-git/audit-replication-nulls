"""Figure 1: the protocol, drawn rather than generated.

Every label corresponds one-to-one with a named element in the protocol section.
Drawing rather than generating removes the risk that an image model inserts
symbols the paper does not define.

The drawing is monochrome and on a grid: boxes in a column share x and width,
boxes in a row share y and height, connectors run only through the corridors
between them, and two line styles carry two meanings stated in the legend.
"""
from __future__ import annotations

import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle



plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 13,
})

FIG = pathlib.Path(__file__).resolve().parents[2] / "output"
FIG.mkdir(parents=True, exist_ok=True)

EDGE, INK = "black", "black"
X0, COLW, GAP = 2.0, 23.0, 5.0
COL = [X0 + i * (COLW + GAP) for i in range(4)]
ROWH = 8.0

fig, ax = plt.subplots(figsize=(11.0, 7.0))
ax.set_xlim(-6, 114)
ax.set_ylim(0, 92)
ax.axis("off")

ROW_A = 80.0
ROW_B1, ROW_B2, ROW_B3 = 58.0, 44.0, 30.0
ROW_C = 11.0


def lane(label, ytop, ybot):
    ax.plot([-3.5, -3.5], [ybot, ytop], color=INK, linewidth=0.9)
    ax.text(-5.0, (ytop + ybot) / 2, label, rotation=90, ha="center",
            va="center", fontsize=12)


def box(col, ycent, text, span=1, fs=12, h=ROWH):
    x = COL[col]
    w = COLW * span + GAP * (span - 1)
    ax.add_patch(Rectangle((x, ycent - h / 2), w, h, facecolor="white",
                           edgecolor=EDGE, linewidth=0.9, zorder=2))
    ax.text(x + w / 2, ycent, text, ha="center", va="center", fontsize=fs,
            linespacing=1.5, zorder=3)
    return (x, x + w, ycent, h)


def arrow(p0, p1, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=1.0, color=INK,
        zorder=4, linestyle=(0, (4, 2.5)) if dashed else "solid",
        shrinkA=1.0, shrinkB=1.0))


def route(points, dashed=False):
    ls = (0, (4, 2.5)) if dashed else "solid"
    ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
            color=INK, linewidth=1.0, linestyle=ls, zorder=4,
            solid_capstyle="butt")
    arrow(points[-2], points[-1], dashed=dashed)


def hlink(a, b):
    arrow((a[1], a[2]), (b[0], b[2]))


def vlink(a, b):
    xc = (a[0] + a[1]) / 2
    arrow((xc, a[2] - a[3] / 2), (xc, b[2] + b[3] / 2))


# ---- band A: the external evidence the criteria come from
lane("A  External evidence", ROW_A + 6.0, ROW_A - 6.0)
a1 = box(0, ROW_A, "Live platform")
a2 = box(1, ROW_A, "Intervention $T$")
a3 = box(2, ROW_A, "Directional finding $d$")
a4 = box(3, ROW_A, "The audit's own\ncontrol condition")
hlink(a1, a2)
hlink(a2, a3)
hlink(a3, a4)

# ---- band B: the procedure
lane("B  Validation procedure", ROW_B1 + 6.0, ROW_B3 - 6.0)
b1 = box(0, ROW_B1, "1  Fix the direction\n2  Transplant the control")
b2 = box(1, ROW_B1, "3  Null-agent test\n4  Calibrated null")
b3 = box(2, ROW_B1, "6  Design sensitivity\nand split replication")
b4 = box(3, ROW_B1, "Run the simulation")
b5 = box(3, ROW_B2, "7  Delivery check")
b6 = box(3, ROW_B3, "5  Attribution analysis")
hlink(b1, b2)
hlink(b2, b3)
hlink(b3, b4)
vlink(b4, b5)
vlink(b5, b6)

ax.text((b2[0] + b3[1]) / 2, ROW_B1 - ROWH / 2 - 1.8,
        "no generative model required",
        ha="center", va="top", fontsize=11)
ax.text(b5[0] - 1.8, (ROW_B1 + ROW_B2) / 2, "after execution",
        ha="right", va="center", fontsize=11)

CORR1, CORR2 = 72.0, 69.0
xb1 = (b1[0] + b1[1]) / 2
route([((a3[0] + a3[1]) / 2, ROW_A - ROWH / 2), ((a3[0] + a3[1]) / 2, CORR1),
       (xb1 - 3.5, CORR1), (xb1 - 3.5, ROW_B1 + ROWH / 2)], dashed=True)
route([((a4[0] + a4[1]) / 2, ROW_A - ROWH / 2), ((a4[0] + a4[1]) / 2, CORR2),
       (xb1 + 3.5, CORR2), (xb1 + 3.5, ROW_B1 + ROWH / 2)], dashed=True)

# ---- band C: how the verdict is read
lane("C  Verdict", ROW_C + 7.5, ROW_C - 7.5)
labels = [
    ("Pass", "direction agrees,\npower sufficient"),
    ("Inconclusive", "direction stable,\npower insufficient"),
    ("Fail", "direction not stable"),
    ("Void", "treatment not delivered"),
]
cb = []
for i, (lab, note) in enumerate(labels):
    c = box(i, ROW_C, lab, fs=12, h=5.8)
    cb.append(c)
    ax.text((c[0] + c[1]) / 2, ROW_C - 5.0, note, ha="center", va="top",
            fontsize=10.5, linespacing=1.45)

spine = ROW_C + 7.0
arrow(((b6[0] + b6[1]) / 2, ROW_B3 - ROWH / 2), ((b6[0] + b6[1]) / 2, spine + 0.6))
ax.plot([(cb[0][0] + cb[0][1]) / 2, (cb[3][0] + cb[3][1]) / 2], [spine, spine],
        color=INK, linewidth=1.0, zorder=4)
for c in cb:
    arrow(((c[0] + c[1]) / 2, spine), ((c[0] + c[1]) / 2, ROW_C + 2.9))

# ---- legend
ly = 90.0
ax.add_patch(FancyArrowPatch((0.0, ly), (7.0, ly), arrowstyle="-|>",
                             mutation_scale=14, linewidth=1.0, color=INK))
ax.text(8.5, ly, "flow of the procedure", va="center", fontsize=11)
ax.add_patch(FancyArrowPatch((42.0, ly), (49.0, ly), arrowstyle="-|>",
                             mutation_scale=14, linewidth=1.0, color=INK,
                             linestyle=(0, (4, 2.5))))
ax.text(50.5, ly, "criterion constraining a later step",
        va="center", fontsize=11)

fig.tight_layout()
fig.savefig(FIG / "fig1_overview.pdf", bbox_inches="tight")
print("wrote {}/fig1_overview.pdf".format(FIG))
