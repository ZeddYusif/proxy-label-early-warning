"""Step 08. E2: how valid are the proxy labels on OULAD, where the official record (L0) is known?

Command:  python 04_src/08_e2_proxy_validity.py --mode demo   (or --mode real)
Writes:   06_results/<mode>/tables/e2_agreement.csv        (RQ1a, part 1: agreement with L0)
          06_results/<mode>/tables/e2_timing.csv           (RQ1a: proxy onset minus official unregistration day)
          06_results/<mode>/tables/e2_train_proxy_judge_truth.csv   (RQ1a, part 2)
          02_data/processed/<mode>/oulad/e2_lags_long.csv  (for the timing figure)
Next:     python 04_src/09_e3_label_multiplicity.py --mode <same mode>

How to read it:
  agreement: kappa against L0, with the band fixed in the lock (acceptable, weak, not valid).
  timing: negative lag = the proxy signal began before the official unregistration.
  train on proxy, judge on truth: a model trained on each proxy is scored against L0 on the test
  presentation. delta_pr_auc is proxy-trained minus L0-trained. "usable" means the 90% interval
  of that loss stays inside the margin.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import (cohort, jaccard, label_columns, load_config, make_model, parse_mode, processed_dir,
                    ranking_metrics, require_lock, save_table, say, top_mask)
from stats_tools import kappa_band, kappa_from_counts

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
feats = cfg["features"]["shared"]
cap = cfg["capacity"]
B = cfg["bootstrap"]
rng = np.random.default_rng(cfg["cv"]["seed"])
proxies = label_columns("oulad")
labs_all = pd.read_csv(processed_dir(args.mode, "oulad") / "labels.csv")
sens_cols = [c for c in labs_all.columns if "_inact" in c]

# ---------------------------------------------------------------- agreement (main-window cohort)
df = cohort(args.mode, "oulad", cfg["main_window"])
y0 = df["L0"].to_numpy()
rows = []
for c in proxies + sens_cols:
    p = df[c].to_numpy()
    tp, fp = int(((p == 1) & (y0 == 1)).sum()), int(((p == 1) & (y0 == 0)).sum())
    fn, tn = int(((p == 0) & (y0 == 1)).sum()), int(((p == 0) & (y0 == 0)).sum())
    k = kappa_from_counts(tp, fp, fn, tn)
    ks = []
    n = len(p)
    for _ in range(B):
        i = rng.integers(0, n, n)
        pp, yy = p[i], y0[i]
        ks.append(kappa_from_counts(((pp == 1) & (yy == 1)).sum(), ((pp == 1) & (yy == 0)).sum(),
                                    ((pp == 0) & (yy == 1)).sum(), ((pp == 0) & (yy == 0)).sum()))
    rows.append(dict(label=c, n=n, kappa=k, kappa_ci_low=np.nanpercentile(ks, 2.5),
                     kappa_ci_high=np.nanpercentile(ks, 97.5),
                     sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                     ppv=tp / (tp + fp) if tp + fp else np.nan,
                     prevalence_ratio=p.mean() / y0.mean() if y0.mean() else np.nan,
                     band=kappa_band(k, cfg["kappa_bands"]["acceptable"], cfg["kappa_bands"]["weak"]),
                     role="main label" if c in proxies else "sensitivity check"))
save_table(pd.DataFrame(rows), "e2_agreement", args.mode)

# ---------------------------------------------------------------- timing
lags = []
for c in proxies:
    on = f"onset_{c.split('_')[0]}_{c.split('_')[1]}"
    sub = df[(df["L0"] == 1) & (df[c] == 1) & df["unreg_day"].notna() & df[on].notna()]
    for v in (sub[on] - sub["unreg_day"]).to_numpy():
        lags.append(dict(label=c, lag_days=v))
lag_df = pd.DataFrame(lags, columns=["label", "lag_days"])
lag_df.to_csv(processed_dir(args.mode, "oulad") / "e2_lags_long.csv", index=False)
trows = []
for c in proxies:
    v = lag_df.loc[lag_df["label"] == c, "lag_days"]
    trows.append(dict(label=c, n=len(v), median_lag=v.median() if len(v) else np.nan,
                      q1=v.quantile(0.25) if len(v) else np.nan, q3=v.quantile(0.75) if len(v) else np.nan,
                      note="onset unknown without due dates" if (c.endswith("emulated") and len(v) == 0) else ""))
save_table(pd.DataFrame(trows), "e2_timing", args.mode)

# ---------------------------------------------------------------- train on proxy, judge on truth
split = cfg["oulad_split"]
res = []
for T in cfg["windows"]:
    d = cohort(args.mode, "oulad", T)
    tr = d[d["presentation"].isin(split["train"])]
    te = d[d["presentation"].isin(split["test"])]
    y_te = te["L0"].to_numpy()
    scores = {}
    for seed in cfg["seeds"]:
        m0 = make_model("hgb", seed).fit(tr[feats], tr["L0"])
        scores[("L0", seed)] = m0.predict_proba(te[feats])[:, 1]
        for c in proxies:
            m = make_model("hgb", seed).fit(tr[feats], tr[c])
            scores[(c, seed)] = m.predict_proba(te[feats])[:, 1]
    for c in ["L0"] + proxies:
        mets = [ranking_metrics(y_te, scores[(c, s)], cap) for s in cfg["seeds"]]
        jac = [jaccard(top_mask(scores[(c, s)], cap), top_mask(scores[("L0", s)], cap)) for s in cfg["seeds"]]
        row = dict(window=T, trained_on=c, n_test=len(y_te), official_rate=y_te.mean(),
                   pr_auc=np.mean([m["pr_auc"] for m in mets]), pr_auc_seed_range=np.ptp([m["pr_auc"] for m in mets]),
                   roc_auc=np.mean([m["roc_auc"] for m in mets]), precision_at=np.mean([m["precision_at"] for m in mets]),
                   recall_at=np.mean([m["recall_at"] for m in mets]), jaccard_with_L0_list=np.mean(jac))
        if c != "L0":
            # one comparison on the seed-averaged risk scores, so the point and its interval match
            avg_c = np.mean([scores[(c, sd)] for sd in cfg["seeds"]], axis=0)
            avg_0 = np.mean([scores[("L0", sd)] for sd in cfg["seeds"]], axis=0)
            point = average_precision_score(y_te, avg_c) - average_precision_score(y_te, avg_0)
            n = len(y_te)
            diffs = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                if y_te[i].min() == y_te[i].max():
                    continue
                diffs.append(average_precision_score(y_te[i], avg_c[i]) - average_precision_score(y_te[i], avg_0[i]))
            lo, hi = np.percentile(diffs, [5, 95])
            mg = cfg["margins"]["proxy_usable_pr_auc_loss"]
            row.update(delta_pr_auc=point, delta_ci90_low=lo, delta_ci90_high=hi,
                       usable="usable" if lo > -mg else ("not usable" if hi < -mg else "inconclusive"))
        res.append(row)
    say(f"  window {T}: trained {1 + len(proxies)} label sets x {len(cfg['seeds'])} seeds")
save_table(pd.DataFrame(res), "e2_train_proxy_judge_truth", args.mode)
say(f"Next: python 04_src/09_e3_label_multiplicity.py --mode {args.mode}")
