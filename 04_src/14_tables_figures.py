"""Step 14. Draw the figures from the result tables. Greyscale, with one olive highlight.

Command:  python 04_src/14_tables_figures.py --mode demo   (or --mode real)
Writes:   06_results/<mode>/figures/*.png (drafts) and *.pdf (for the journal)
Next:     read every table and figure, then write up (Workflow Guide, sections 6.1 to 6.3 explain each one).

Demo figures carry the watermark "SYNTHETIC DEMO DATA - not a result". Real figures do not.
Every figure comes from a table made by code. Never edit a figure or table by hand.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import WATERMARK, load_config, parse_mode, processed_dir, read_table, results_dir, say  # noqa: E402

args = parse_mode(__doc__)
cfg = load_config()
OLIVE, DARK, MID, LIGHT = "#556B2F", "#222222", "#7F7F7F", "#BFBFBF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": DARK, "axes.linewidth": 0.8})
figdir = results_dir(args.mode, "figures")


def finish(fig, name):
    if args.mode == "demo":
        fig.text(0.5, 0.5, WATERMARK, ha="center", va="center", rotation=20, fontsize=16, color="#D0D0D0", zorder=0,
                 alpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    say(f"  wrote figures/{name}.png and .pdf")


# Figure 1: kappa of each proxy against L0
a = read_table("e2_agreement", args.mode)
a = a[a["role"] == "main label"]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
yy = np.arange(len(a))
ax.errorbar(a["kappa"], yy, xerr=[a["kappa"] - a["kappa_ci_low"], a["kappa_ci_high"] - a["kappa"]],
            fmt="o", color=DARK, ecolor=MID, capsize=2)
ax.axvline(cfg["kappa_bands"]["acceptable"], color=OLIVE, lw=1.2, ls="--", label="acceptable (fixed in lock)")
ax.axvline(cfg["kappa_bands"]["weak"], color=MID, lw=0.8, ls=":", label="weak")
ax.set_yticks(yy, a["label"])
ax.set_xlabel("Cohen's kappa with official withdrawal (L0), 95% interval")
ax.set_xlim(-0.05, 1.0)
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_title("E2. Agreement of proxy labels with official withdrawal (OULAD)", fontsize=9, loc="left")
finish(fig, "fig_e2_kappa")

# Figure 2: timing lags
lags = pd.read_csv(processed_dir(args.mode, "oulad") / "e2_lags_long.csv")
labs = [lab for lab in lags["label"].unique() if (lags["label"] == lab).sum() > 0]
if labs:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    horiz = {"orientation": "horizontal"} if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 10) else {"vert": False}
    ax.boxplot([lags.loc[lags["label"] == lab, "lag_days"] for lab in labs], widths=0.5, **horiz,
               medianprops=dict(color=OLIVE, lw=1.5), flierprops=dict(marker=".", markersize=2, markeredgecolor=MID))
    ax.set_yticks(range(1, len(labs) + 1), labs)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_xlabel("Proxy onset minus official unregistration day (negative = proxy signal came first)")
    ax.set_title("E2. When does each proxy signal begin? (OULAD, withdrawn students)", fontsize=9, loc="left")
    finish(fig, "fig_e2_timing")

# Figure 3: Jaccard heatmap of top-20% lists (KNUST)
pairs = read_table("e3_overlap_pairs", args.mode)
labels = sorted(set(pairs["label_a"]) | set(pairs["label_b"]))
M = np.eye(len(labels))
for _, r in pairs.iterrows():
    i, j = labels.index(r["label_a"]), labels.index(r["label_b"])
    M[i, j] = M[j, i] = r["jaccard_top"]
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.6 else DARK)
ax.set_xticks(range(len(labels)), labels)
ax.set_yticks(range(len(labels)), labels)
fig.colorbar(im, ax=ax, fraction=0.046, label="Jaccard overlap of top-20% lists")
ax.set_title("E3. Same students, different alerts (KNUST)", fontsize=9, loc="left")
finish(fig, "fig_e3_jaccard")

# Figure 4: pre-onset share by window
p = read_table("e4_pre_onset_share", args.mode).dropna(subset=["pre_onset_share"])
fig, ax = plt.subplots(figsize=(6.2, 3.0))
greys = [DARK, MID, LIGHT, "#4D4D4D", "#A6A6A6"]
for k, ((ds, lab), g) in enumerate(p.groupby(["dataset", "label"])):
    if lab.startswith("L1"):
        ax.plot(g["window"], g["pre_onset_share"], marker="o", color=OLIVE if ds == "knust" else greys[k % 5],
                lw=1.8 if ds == "knust" else 1.0, label=f"{ds} {lab}")
ax.set_xlabel("Prediction day")
ax.set_ylabel("Share of L1 dropouts already\ninactive before the prediction day")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7)
ax.set_title("E4. How much of the label is already visible at prediction time?", fontsize=9, loc="left")
finish(fig, "fig_e4_pre_onset")

# Figure 5: equivalence forest plot
eq = read_table("e5_equivalence", args.mode)
eq = eq[eq["metric"] == "roc_auc"].reset_index(drop=True)
try:
    e6 = read_table("e6_tabpfn_equivalence", args.mode)
    if "metric" in e6:
        eq = pd.concat([eq, e6[e6["metric"] == "roc_auc"]], ignore_index=True)
except SystemExit:
    pass
fig, ax = plt.subplots(figsize=(6.6, 0.45 * len(eq) + 1.2))
mgn = cfg["margins"]["roc_auc"]
ax.axvspan(-mgn, mgn, color=OLIVE, alpha=0.15, lw=0, label=f"equivalence margin (plus or minus {mgn})")
ax.axvline(0, color=DARK, lw=0.8)
for i, r in eq.iterrows():
    ax.plot([r["ci90_low"], r["ci90_high"]], [i, i], color=DARK, lw=2)
    ax.plot([r["ci95_low"], r["ci95_high"]], [i, i], color=MID, lw=0.8)
    ax.plot(r["mean_diff"], i, "o", color=DARK, ms=4)
    ax.text(max(r["ci95_high"], mgn) + 0.004, i, r["verdict"], va="center", fontsize=7)
ax.set_yticks(range(len(eq)), [c.replace(" minus ", "\nminus ") for c in eq["comparison"]], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Difference in ROC-AUC (thick: 90% interval, thin: 95% interval)")
ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.set_title("E5 and E6. Equivalent, different or inconclusive? (KNUST, L1, main window)", fontsize=9, loc="left")
finish(fig, "fig_e5_equivalence")
say("Done. Read the Workflow Guide, sections 6.1 to 6.3, to interpret each output.")
