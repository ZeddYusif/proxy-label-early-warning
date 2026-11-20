"""Step 11. E5: does the OULAD pipeline transfer to KNUST, and is each arm equivalent to local training?

Command:  python 04_src/11_e5_transfer_equivalence.py --mode demo   (or --mode real)
Writes:   06_results/<mode>/tables/e5_per_fold.csv        (every fold of every arm; kept for E6 and the supplement)
          06_results/<mode>/tables/e5_arms_summary.csv    (A zero-shot, B local, C stacking, D pooled)
          06_results/<mode>/tables/e5_equivalence.csv     (each comparison ends: equivalent, different or inconclusive)
          06_results/<mode>/tables/e5_small_sample.csv    (20 draws of 300 students)
Next:     python 04_src/12_e6_tabpfn.py --mode <same mode>

Arms (as in your draft): A = model trained on all OULAD (official label), applied to KNUST unchanged.
B = trained and tested on KNUST only. C = B plus the OULAD model's log-odds as one extra feature.
D = OULAD rows plus KNUST training rows, KNUST rows weighted five times. KNUST outcome = L1.
Fold comparisons use the Nadeau and Bengio corrected variance. Zero-shot A against local B uses a
paired bootstrap over students. Margins come from the lock.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from common import cohort, load_config, make_model, parse_mode, ranking_metrics, require_lock, save_table, say
from stats_tools import bootstrap_compare, corrected_compare

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
feats = cfg["features"]["shared"]
cap = cfg["capacity"]
cv = cfg["cv"]
T = cfg["main_window"]
ou = cohort(args.mode, "oulad", T)
kn = cohort(args.mode, "knust", T)
y = kn["L1"].to_numpy()
seed = cv["seed"]


def fit(kind, X, yy, w=None, s=seed):
    m = make_model(kind, s)
    if w is None:
        return m.fit(X, yy)
    if kind == "hgb":
        return m.fit(X, yy, histgradientboostingclassifier__sample_weight=w)
    return m.fit(X, yy, logisticregression__sample_weight=w)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


zero = {k: fit(k, ou[feats], ou["L0"]) for k in ["hgb", "logreg"]}
a_scores = {k: zero[k].predict_proba(kn[feats])[:, 1] for k in zero}
kn_stack = kn[feats].assign(oulad_logodds=logit(a_scores["hgb"]))

rows = []
oof_b = np.full((cv["n_repeats"], len(kn)), np.nan)
rskf = RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=seed)
for i, (tr, te) in enumerate(rskf.split(kn[feats], y)):
    r, f = divmod(i, cv["n_splits"])
    for kind in ["hgb", "logreg"]:
        arms = {
            "B local": (kn[feats].iloc[tr], y[tr], None, kn[feats].iloc[te]),
            "C stacking": (kn_stack.iloc[tr], y[tr], None, kn_stack.iloc[te]),
            "D pooled": (pd.concat([ou[feats], kn[feats].iloc[tr]]),
                         np.concatenate([ou["L0"].to_numpy(), y[tr]]),
                         np.concatenate([np.ones(len(ou)), np.full(len(tr), 5.0)]), kn[feats].iloc[te]),
        }
        for arm, (Xtr, ytr, w, Xte) in arms.items():
            s = fit(kind, Xtr, ytr, w, seed + r).predict_proba(Xte)[:, 1]
            if arm == "B local" and kind == "hgb":
                oof_b[r, te] = s
            met = ranking_metrics(y[te], s, cap)
            rows.append(dict(arm=arm, model=kind, repeat=r, fold=f, n_train=len(tr), n_test=len(te),
                             dropouts_in_test=int(y[te].sum()), **met))
    if f == cv["n_splits"] - 1:
        say(f"  repeat {r + 1}/{cv['n_repeats']} done")
per_fold = pd.DataFrame(rows)
save_table(per_fold, "e5_per_fold", args.mode)

summ = []
for kind in ["hgb", "logreg"]:
    a = ranking_metrics(y, a_scores[kind], cap)
    summ.append(dict(arm="A zero-shot", model=kind, pr_auc=a["pr_auc"], roc_auc=a["roc_auc"], recall_at=a["recall_at"],
                     prevalence=y.mean()))
for (arm, kind), g in per_fold.groupby(["arm", "model"]):
    summ.append(dict(arm=arm, model=kind, pr_auc=g["pr_auc"].mean(), roc_auc=g["roc_auc"].mean(),
                     recall_at=g["recall_at"].mean(), prevalence=y.mean()))
save_table(pd.DataFrame(summ), "e5_arms_summary", args.mode)

# ---------------------------------------------------------------- equivalence
mg = {"pr_auc": cfg["margins"]["pr_auc"], "roc_auc": cfg["margins"]["roc_auc"], "recall_at": cfg["margins"]["recall_at_capacity"]}
comps = [(("C stacking", "hgb"), ("B local", "hgb")), (("D pooled", "hgb"), ("B local", "hgb")),
         (("B local", "logreg"), ("B local", "hgb")), (("C stacking", "logreg"), ("B local", "logreg")),
         (("D pooled", "logreg"), ("B local", "logreg"))]
eq = []
for (a1, m1), (a2, m2) in comps:
    g1 = per_fold[(per_fold.arm == a1) & (per_fold.model == m1)].sort_values(["repeat", "fold"])
    g2 = per_fold[(per_fold.arm == a2) & (per_fold.model == m2)].sort_values(["repeat", "fold"])
    for met, margin in mg.items():
        r = corrected_compare(g1[met].to_numpy() - g2[met].to_numpy(), g1["n_train"].mean(), g1["n_test"].mean(), margin)
        eq.append(dict(comparison=f"{a1} ({m1}) minus {a2} ({m2})", metric=met, margin=margin, method="corrected resampled t", **r))
# A against B: paired bootstrap over students, B scored by its out-of-fold risk averaged over repeats
b_mean = np.nanmean(oof_b, axis=0)
rng = np.random.default_rng(seed)
da, db = [], []
n = len(y)
for _ in range(cfg["bootstrap"]):
    i = rng.integers(0, n, n)
    if y[i].min() == y[i].max():
        continue
    da.append(roc_auc_score(y[i], a_scores["hgb"][i]))
    db.append(roc_auc_score(y[i], b_mean[i]))
r = bootstrap_compare(da, db, mg["roc_auc"])
eq.append(dict(comparison="A zero-shot (hgb) minus B local (hgb)", metric="roc_auc", margin=mg["roc_auc"],
               method="paired bootstrap over students", **r))
save_table(pd.DataFrame(eq), "e5_equivalence", args.mode)

# ---------------------------------------------------------------- small-sample draws
ss = cfg["small_sample"]
draw_rows = []
for dnum in range(ss["n_draws"]):
    rng_d = np.random.default_rng(1000 + dnum)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = int(round(ss["size"] * y.mean()))
    idx = np.concatenate([rng_d.choice(pos, k, replace=False), rng_d.choice(neg, ss["size"] - k, replace=False)])
    sub, ys = kn.iloc[idx], y[idx]
    subs = kn_stack.iloc[idx]
    out = {"A zero-shot": a_scores["hgb"][idx]}
    for arm in ["B local", "C stacking", "D pooled"]:
        out[arm] = np.full(len(idx), np.nan)
    skf = StratifiedKFold(n_splits=cv["n_splits"], shuffle=True, random_state=dnum)
    for tr, te in skf.split(sub[feats], ys):
        out["B local"][te] = fit("hgb", sub[feats].iloc[tr], ys[tr]).predict_proba(sub[feats].iloc[te])[:, 1]
        out["C stacking"][te] = fit("hgb", subs.iloc[tr], ys[tr]).predict_proba(subs.iloc[te])[:, 1]
        Xd = pd.concat([ou[feats], sub[feats].iloc[tr]])
        yd = np.concatenate([ou["L0"].to_numpy(), ys[tr]])
        wd = np.concatenate([np.ones(len(ou)), np.full(len(tr), 5.0)])
        out["D pooled"][te] = fit("hgb", Xd, yd, wd).predict_proba(sub[feats].iloc[te])[:, 1]
    for arm, s in out.items():
        met = ranking_metrics(ys, s, cap)
        draw_rows.append(dict(draw=dnum, arm=arm, roc_auc=met["roc_auc"], pr_auc=met["pr_auc"]))
dr = pd.DataFrame(draw_rows)
small = dr.groupby("arm").agg(draws=("draw", "nunique"), roc_auc_mean=("roc_auc", "mean"), roc_auc_min=("roc_auc", "min"),
                              roc_auc_max=("roc_auc", "max"), pr_auc_mean=("pr_auc", "mean")).reset_index()
save_table(small, "e5_small_sample", args.mode)
say(f"Next: python 04_src/12_e6_tabpfn.py --mode {args.mode}")
