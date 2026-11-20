"""Step 09. E3: on KNUST data, how much does the alert list change with the proxy label?

Command:  python 04_src/09_e3_label_multiplicity.py --mode demo   (or --mode real)
Writes:   06_results/<mode>/tables/e3_performance.csv     (per label and model: PR-AUC, lift, recall at capacity)
          06_results/<mode>/tables/e3_overlap_pairs.csv   (Jaccard of top-20% lists and Spearman of risks, per label pair)
          06_results/<mode>/tables/e3_flip_rate.csv       (share of students flagged under some labels but not all)
          06_results/<mode>/tables/e3_channel_audit.csv   (alert and flip rate by main access channel)
Next:     python 04_src/10_e4_circularity.py --mode <same mode>

Design: repeated stratified 5-fold cross-validation, 10 repeats, at the main window.
Each repeat gives every student one out-of-fold risk score. Lists are compared repeat by repeat.
PR-AUC lift = PR-AUC divided by prevalence, because prevalence differs between labels.
Groups with fewer than 10 students are never reported.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (cohort, jaccard, label_columns, load_config, make_model, parse_mode, ranking_metrics,
                    require_lock, save_table, say, top_mask)

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
feats = cfg["features"]["shared"]
cap = cfg["capacity"]
cv = cfg["cv"]
labels = label_columns("knust")
df = cohort(args.mode, "knust", cfg["main_window"])
X = df[feats]
n = len(df)
oof = {}
perf = []
for lab in labels:
    y = df[lab].to_numpy()
    for kind in ["hgb", "logreg"]:
        scores = np.full((cv["n_repeats"], n), np.nan)
        rskf = RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=cv["seed"])
        for i, (tr, te) in enumerate(rskf.split(X, y)):
            r = i // cv["n_splits"]
            m = make_model(kind, cv["seed"] + r).fit(X.iloc[tr], y[tr])
            scores[r, te] = m.predict_proba(X.iloc[te])[:, 1]
        oof[(lab, kind)] = scores
        reps = [ranking_metrics(y, scores[r], cap) for r in range(cv["n_repeats"])]
        pr = np.array([m["pr_auc"] for m in reps])
        perf.append(dict(label=lab, model=kind, n=n, prevalence=y.mean(), pr_auc=pr.mean(),
                         pr_auc_low=np.percentile(pr, 2.5), pr_auc_high=np.percentile(pr, 97.5),
                         pr_auc_lift=pr.mean() / y.mean() if y.mean() else np.nan,
                         roc_auc=np.mean([m["roc_auc"] for m in reps]),
                         recall_at=np.mean([m["recall_at"] for m in reps]),
                         precision_at=np.mean([m["precision_at"] for m in reps])))
    say(f"  {lab}: prevalence {df[lab].mean():.1%}, hgb and logreg done")
save_table(pd.DataFrame(perf), "e3_performance", args.mode)

# ---------------------------------------------------------------- overlap and flips (hgb)
pairs = []
for a, b in combinations(labels, 2):
    jac, rho = [], []
    for r in range(cv["n_repeats"]):
        sa, sb = oof[(a, "hgb")][r], oof[(b, "hgb")][r]
        jac.append(jaccard(top_mask(sa, cap, r), top_mask(sb, cap, r)))
        rho.append(spearmanr(sa, sb).statistic)
    pairs.append(dict(label_a=a, label_b=b, jaccard_top=np.mean(jac), jaccard_low=np.min(jac),
                      jaccard_high=np.max(jac), spearman_risk=np.mean(rho)))
save_table(pd.DataFrame(pairs), "e3_overlap_pairs", args.mode)

flags = np.stack([[top_mask(oof[(lab, "hgb")][r], cap, r) for r in range(cv["n_repeats"])] for lab in labels])  # L x R x n
any_f = flags.any(axis=0)
all_f = flags.all(axis=0)
flip = (any_f & ~all_f)                                                                                    # R x n
save_table(pd.DataFrame([dict(labels=", ".join(labels), n=n, flip_rate=flip.mean(axis=1).mean(),
                              flagged_by_any=any_f.mean(axis=1).mean(), flagged_by_all=all_f.mean(axis=1).mean())]),
           "e3_flip_rate", args.mode)

# ---------------------------------------------------------------- access-channel audit
l1_flags = flags[labels.index("L1")]
rows = []
ch = df["channel"].fillna("unknown").to_numpy()
for c in sorted(set(ch)):
    idx = ch == c
    pos = int(df.loc[idx, "L1"].sum())
    rows.append(dict(channel=c, n=int(idx.sum()), L1_positives=pos, L1_prevalence=df.loc[idx, "L1"].mean(),
                     alert_rate=l1_flags[:, idx].mean(), flip_rate=flip[:, idx].mean()))
ca = pd.DataFrame(rows)
ca = ca[ca["n"] >= 10]
enough = (ca["L1_positives"] >= cfg["channel_audit"]["min_positives_per_channel"]).sum() >= 2
ca["reportable"] = "yes" if enough else "no: fewer than two channels with enough L1 positives"
save_table(ca, "e3_channel_audit", args.mode)
say(f"Next: python 04_src/10_e4_circularity.py --mode {args.mode}")
