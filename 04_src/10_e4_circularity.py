"""Step 10. E4: how much of the gain from later windows comes from features and labels that share a signal?

Command:  python 04_src/10_e4_circularity.py --mode demo   (or --mode real)
Writes:   06_results/<mode>/tables/e4_pre_onset_share.csv   (share of proxy dropouts already inactive before day T)
          06_results/<mode>/tables/e4_knust_active_and_ablation.csv   (all students, students active in the last
                                                                       7 days, and the model without recency features)
          06_results/<mode>/tables/e4_oulad_gain_vs_official.csv      (gain from day 14 to day 42, proxy against L0)
Next:     python 04_src/11_e5_transfer_equivalence.py --mode <same mode>

Reading it: a high pre-onset share means the label was already visible in the features.
The adviser can still help only students who were active recently, so their row matters most.
On OULAD, L0 does not come from the logs. Extra gain under a proxy (proxy gain minus L0 gain)
estimates circularity. ROC-AUC is used for gains because it does not move with prevalence.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (cohort, label_columns, load_config, make_model, parse_mode, ranking_metrics,
                    require_lock, save_table, say)

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
feats = cfg["features"]["shared"]
cap = cfg["capacity"]
cv = cfg["cv"]

# ---------------------------------------------------------------- pre-onset share
rows = []
for ds in ["oulad", "knust"]:
    for T in cfg["windows"]:
        d = cohort(args.mode, ds, T)
        for c in label_columns(ds):
            on = f"onset_{c}"
            if on not in d:
                continue
            pos = d[d[c] == 1]
            known = pos[on].notna()
            rows.append(dict(dataset=ds, window=T, label=c, positives=len(pos),
                             onset_known=int(known.sum()),
                             pre_onset_share=(pos.loc[known, on] < T).mean() if known.any() else np.nan))
save_table(pd.DataFrame(rows), "e4_pre_onset_share", args.mode)

# ---------------------------------------------------------------- KNUST: active students and ablation
no_recency = [f for f in feats if f not in cfg["features"]["recency"]]
out = []
for T in cfg["windows"]:
    d = cohort(args.mode, "knust", T)
    y = d["L1"].to_numpy()
    active = d["active_last7"].to_numpy() == 1 if "active_last7" in d else np.ones(len(d), bool)
    for name, cols in [("all features", feats), ("without recency features", no_recency)]:
        scores = np.full((cv["n_repeats"], len(d)), np.nan)
        rskf = RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=cv["seed"])
        for i, (tr, te) in enumerate(rskf.split(d[cols], y)):
            r = i // cv["n_splits"]
            m = make_model("hgb", cv["seed"] + r).fit(d[cols].iloc[tr], y[tr])
            scores[r, te] = m.predict_proba(d[cols].iloc[te])[:, 1]
        for grp, idx in [("all students", np.ones(len(d), bool)), ("active in last 7 days", active)]:
            if idx.sum() < 10 or y[idx].sum() < 5:
                out.append(dict(window=T, features=name, group=grp, n=int(idx.sum()), note="too few to report"))
                continue
            reps = [ranking_metrics(y[idx], scores[r][idx], cap) for r in range(cv["n_repeats"])]
            out.append(dict(window=T, features=name, group=grp, n=int(idx.sum()), prevalence=y[idx].mean(),
                            pr_auc=np.mean([m["pr_auc"] for m in reps]), roc_auc=np.mean([m["roc_auc"] for m in reps]),
                            recall_at=np.mean([m["recall_at"] for m in reps]), note=""))
    say(f"  KNUST window {T} done")
save_table(pd.DataFrame(out), "e4_knust_active_and_ablation", args.mode)

# ---------------------------------------------------------------- OULAD: proxy gain against official gain
split = cfg["oulad_split"]
seed = cfg["seeds"][0]
roc = {}
for T in [min(cfg["windows"]), max(cfg["windows"])]:
    d = cohort(args.mode, "oulad", T)
    tr = d[d["presentation"].isin(split["train"])]
    te = d[d["presentation"].isin(split["test"])]
    for c in ["L0"] + label_columns("oulad"):
        m = make_model("hgb", seed).fit(tr[feats], tr[c])
        roc[(c, T)] = ranking_metrics(te[c].to_numpy(), m.predict_proba(te[feats])[:, 1], cap)["roc_auc"]
lo, hi = min(cfg["windows"]), max(cfg["windows"])
g0 = roc[("L0", hi)] - roc[("L0", lo)]
rows = [dict(label=c, roc_auc_early=roc[(c, lo)], roc_auc_late=roc[(c, hi)], gain=roc[(c, hi)] - roc[(c, lo)],
             gain_official=g0, extra_gain=(roc[(c, hi)] - roc[(c, lo)]) - g0)
        for c in ["L0"] + label_columns("oulad")]
save_table(pd.DataFrame(rows), "e4_oulad_gain_vs_official", args.mode)
say(f"Next: python 04_src/11_e5_transfer_equivalence.py --mode {args.mode}")
