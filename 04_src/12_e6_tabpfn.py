"""Step 12. E6: a current-method comparator, TabPFN, added after the freeze. It never replaces HGB.

Command:  python 04_src/12_e6_tabpfn.py --mode demo                  (uses a stand-in model, see below)
          python 04_src/12_e6_tabpfn.py --mode real                  (needs: pip install -r requirements-extra.txt)
          python 04_src/12_e6_tabpfn.py --mode real --backend tabpfn  (fail loudly if TabPFN cannot load)
Writes:   06_results/<mode>/tables/e6_tabpfn_folds.csv
          06_results/<mode>/tables/e6_tabpfn_equivalence.csv
Next:     python 04_src/13_e7_registrar_check.py --mode <same mode>

Same folds as E5 (same seed), KNUST outcome L1, main window, default settings, no tuning.
Only the first `tabpfn.folds_used` folds run, to keep laptop CPU time modest (ASSUMPTION: about
10 to 30 seconds per fold on a laptop CPU. Time your first fold and note it in the falsification log).
Demo mode uses a stand-in (extra-trees) labelled MOCK, because the demo must run offline and free.
The first real run downloads the TabPFN model weights from Hugging Face. They are gated (tested
19 November 2026 with tabpfn 10.2.0): accept the terms on the model page named in the error message,
then run  hf auth login  once (Workflow Guide, section 5.3). Report the package version and the checkpoint name the
log prints. Run it on your laptop. KNUST data never goes to Colab.
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from common import cohort, load_config, parse_mode, ranking_metrics, read_table, require_lock, save_table, say
from stats_tools import corrected_compare

args = parse_mode(__doc__, lambda p: p.add_argument("--backend", choices=["auto", "tabpfn", "mock"], default=None))
require_lock(args.mode)
cfg = load_config()
tc = cfg["tabpfn"]
backend = args.backend or ("mock" if args.mode == "demo" else tc["backend"])


def make_tabpfn():
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=tc["device"])


def make_mock():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                                                                random_state=0, n_jobs=-1))


if backend in ("auto", "tabpfn"):
    try:
        import tabpfn
        rng0 = np.random.default_rng(0)
        Xt = rng0.random((40, 3))
        make_tabpfn().fit(Xt, (Xt[:, 0] > 0.5).astype(int))       # loads the weights once, or fails here
        model_name, factory = f"tabpfn {tabpfn.__version__}", make_tabpfn
    except Exception as e:  # noqa: BLE001
        if backend == "tabpfn":
            raise
        first = (str(e).strip().splitlines() or [""])[0][:160]
        say(f"TabPFN could not load ({type(e).__name__}: {first}). E6 skipped. See the Workflow Guide, section 5.3.")
        save_table(pd.DataFrame([dict(status="not run", reason=str(e)[:200])]), "e6_tabpfn_equivalence", args.mode)
        raise SystemExit
else:
    model_name, factory = "MOCK stand-in (not TabPFN)", make_mock

feats = cfg["features"]["shared"]
cv = cfg["cv"]
cap = cfg["capacity"]
kn = cohort(args.mode, "knust", cfg["main_window"])
y = kn["L1"].to_numpy()
rskf = RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=cv["seed"])
rows = []
for i, (tr, te) in enumerate(rskf.split(kn[feats], y)):
    if i >= tc["folds_used"]:
        break
    r, f = divmod(i, cv["n_splits"])
    t0 = time.time()
    m = factory().fit(kn[feats].iloc[tr].to_numpy(), y[tr])
    s = m.predict_proba(kn[feats].iloc[te].to_numpy())[:, 1]
    rows.append(dict(model=model_name, repeat=r, fold=f, n_train=len(tr), n_test=len(te), seconds=time.time() - t0,
                     **ranking_metrics(y[te], s, cap)))
    say(f"  fold {i + 1}/{tc['folds_used']}: {rows[-1]['seconds']:.1f} s")
folds = pd.DataFrame(rows)
save_table(folds, "e6_tabpfn_folds", args.mode)

hgb = read_table("e5_per_fold", args.mode)
hgb = hgb[(hgb.arm == "B local") & (hgb.model == "hgb")]
m = folds.merge(hgb, on=["repeat", "fold"], suffixes=("_new", "_hgb"))
mg = {"pr_auc": cfg["margins"]["pr_auc"], "roc_auc": cfg["margins"]["roc_auc"], "recall_at": cfg["margins"]["recall_at_capacity"]}
eq = []
for met, margin in mg.items():
    r = corrected_compare(m[f"{met}_new"] - m[f"{met}_hgb"], m["n_train_new"].mean(), m["n_test_new"].mean(), margin)
    eq.append(dict(comparison=f"{model_name} minus B local (hgb)", metric=met, margin=margin,
                   added_after_freeze="yes", **r))
save_table(pd.DataFrame(eq), "e6_tabpfn_equivalence", args.mode)
say(f"Next: python 04_src/13_e7_registrar_check.py --mode {args.mode}")
