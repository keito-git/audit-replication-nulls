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
X0, COLW, GAP = 2.0, 18.0, 4.0
COL = [X0 + i * (COLW + GAP) for i in range(5)]
ROWH = 8.0

fig, ax = plt.subplots(figsize=(11.6, 7.4))
ax.set_xlim(-6, 114)
ax.set_ylim(-2, 92)
ax.axis("off")

ROW_A = 80.0
ROW_B1, ROW_B2 = 56.0, 42.0
ROW_C = 24.0
ROW_D = 8.0


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
a1 = box(0, ROW_A, "Target system $\\mathcal{S}$")
a2 = box(1, ROW_A, "Intervention $T$")
a3 = box(2, ROW_A, "Direction $d$ of\nthe observable $\\psi$")
a4 = box(3, ROW_A, "The audit's own\ncontrol condition")
hlink(a1, a2)
hlink(a2, a3)
hlink(a3, a4)

# ---- band B: the procedure
lane("B  Validation procedure", ROW_B1 + 6.0, ROW_B2 - 6.0)
b1 = box(0, ROW_B1, "1  Fix the\ndirection")
b2 = box(1, ROW_B1, "2  Transplant\nthe control")
# the Japanese label is the widest in the figure, so it takes a smaller size
b3 = box(2, ROW_B1, "3  Null-agent test\n4  Calibrated null", fs=12)
b4 = box(3, ROW_B1, "6  Design sensitivity\nand split replication", fs=12)
b5 = box(4, ROW_B1, "Run the\nsimulation")
b6 = box(4, ROW_B2, "7  Delivery check")
hlink(b1, b2)
hlink(b2, b3)
hlink(b3, b4)
hlink(b4, b5)
vlink(b5, b6)

ax.text((b3[0] + b4[1]) / 2, ROW_B1 - ROWH / 2 - 1.8,
        "no generative model required",
        ha="center", va="top", fontsize=11)
ax.text(b6[0] - 1.8, ROW_B2 + ROWH / 2 + 1.2, "after execution",
        ha="right", va="center", fontsize=11)

# each criterion constrains the element that uses it, so the two connectors land
# on different boxes and are routed through corridors of their own
CORR1, CORR2 = 71.0, 67.0
route([((a3[0] + a3[1]) / 2, ROW_A - ROWH / 2), ((a3[0] + a3[1]) / 2, CORR1),
       ((b1[0] + b1[1]) / 2, CORR1), ((b1[0] + b1[1]) / 2, ROW_B1 + ROWH / 2)],
      dashed=True)
route([((a4[0] + a4[1]) / 2, ROW_A - ROWH / 2), ((a4[0] + a4[1]) / 2, CORR2),
       ((b2[0] + b2[1]) / 2, CORR2), ((b2[0] + b2[1]) / 2, ROW_B1 + ROWH / 2)],
      dashed=True)

# ---- band C: the verdict the delivery check and the criteria produce
lane("C  Verdict", ROW_C + 7.0, ROW_C - 7.0)
labels = [
    ("Pass", "every pass\ncriterion met"),
    ("Inconclusive", "amplification and stable\ndirection, power short"),
    ("Fail", "no amplification, or\ndirection not stable"),
    ("Void", "treatment not delivered"),
]
cb = []
for i, (lab, note) in enumerate(labels):
    c = box(i, ROW_C, lab, fs=12, h=5.6)
    cb.append(c)
    ax.text((c[0] + c[1]) / 2, ROW_C - 4.8, note, ha="center", va="top",
            fontsize=10.5, linespacing=1.45)

# the delivery check and the pass criteria are what decide the verdict, so the
# spine that feeds the four outcomes starts there
spine = ROW_C + 7.0
x6 = (b6[0] + b6[1]) / 2
xl = (cb[0][0] + cb[0][1]) / 2
ax.plot([x6, x6], [ROW_B2 - ROWH / 2, spine], color=INK, linewidth=1.0, zorder=4)
ax.plot([xl, x6], [spine, spine], color=INK, linewidth=1.0, zorder=4)
for c in cb:
    arrow(((c[0] + c[1]) / 2, spine), ((c[0] + c[1]) / 2, ROW_C + 2.8))
ax.text((xl + x6) / 2, spine + 1.4,
        "separated on delivery and on the pass criteria",
        ha="center", va="bottom", fontsize=11)

# ---- band D: what a pass is then subjected to
lane("D  Interpretation", ROW_D + 5.0, ROW_D - 5.0)
d1 = box(0, ROW_D, "5  Attribution analysis", span=2)
xd = (COL[0] + COLW + COL[1]) / 2
route([(cb[0][1], ROW_C), (xd, ROW_C), (xd, ROW_D + ROWH / 2)])
ax.text(d1[1] + 2.0, ROW_D,
        "how much of a passing reproduction\nthe agents alone already explain",
        ha="left", va="center", fontsize=11, linespacing=1.45)

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
