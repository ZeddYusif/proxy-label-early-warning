"""Step 13. E7 (optional): the Registrar's near-gold check of the KNUST proxy label L1.

Command:  python 04_src/13_e7_registrar_check.py --mode demo   (practice table from the hidden demo truth)
          python 04_src/13_e7_registrar_check.py --mode real   (needs 02_data/raw/knust/registrar_2x2.csv)
Writes:   06_results/<mode>/tables/e7_registrar_check.csv
Next:     python 04_src/14_tables_figures.py --mode <same mode>

The Registrar draws 100 students with L1 = 1 and 100 with L1 = 0 at random, checks official status,
and returns ONLY aggregate counts in the template 07_docs/templates/registrar_2x2_template.csv.
No identity leaves the Registrar. You never type or edit these counts yourself beyond copying them.
PPV comes from the L1 = 1 sample, NPV from the L1 = 0 sample. Sensitivity and specificity are
re-weighted with the L1 prevalence in the full cohort (shown as derived values).
"""
import numpy as np
import pandas as pd

from common import ROOT, cohort, load_config, parse_mode, require_lock, save_table, say, wilson

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
kn = cohort(args.mode, "knust", cfg["main_window"])
prev = kn["L1"].mean()

if args.mode == "demo":
    truth = pd.read_csv(ROOT / cfg["paths"]["demo_raw"] / "knust" / "hidden_truth_DEMO_ONLY.csv")
    d = kn.merge(truth, left_on="student_key", right_on="participant_id")
    rng = np.random.default_rng(7)
    rows = []
    for v in [1, 0]:
        s = d[d["L1"] == v].sample(n=min(100, (d["L1"] == v).sum()), random_state=int(rng.integers(1e6)))
        for t in [1, 0]:
            rows.append(dict(proxy_L1=v, registry_withdrawn_or_inactive=t, count=int((s["truly_withdrawn"] == t).sum())))
    table = pd.DataFrame(rows)
else:
    p = ROOT / cfg["paths"]["knust_raw"] / "registrar_2x2.csv"
    if not p.exists():
        say("E7 not run: registrar_2x2.csv not found. Record E7 as pending or dropped (checklist item).")
        save_table(pd.DataFrame([dict(status="not run", reason="no Registrar table")]), "e7_registrar_check", args.mode)
        raise SystemExit
    table = pd.read_csv(p)


def cnt(v, t):
    return int(table[(table.proxy_L1 == v) & (table.registry_withdrawn_or_inactive == t)]["count"].sum())


tp, fp, fn, tn = cnt(1, 1), cnt(1, 0), cnt(0, 1), cnt(0, 0)
ppv = wilson(tp, tp + fp)
npv = wilson(tn, tn + fn)
sens = ppv[0] * prev / (ppv[0] * prev + (1 - npv[0]) * (1 - prev)) if not np.isnan(ppv[0] + npv[0]) else np.nan
spec = npv[0] * (1 - prev) / (npv[0] * (1 - prev) + (1 - ppv[0]) * prev) if not np.isnan(ppv[0] + npv[0]) else np.nan
out = pd.DataFrame([
    dict(measure="PPV of L1", estimate=ppv[0], ci_low=ppv[1], ci_high=ppv[2], n=tp + fp, basis="L1 = 1 sample"),
    dict(measure="NPV of L1", estimate=npv[0], ci_low=npv[1], ci_high=npv[2], n=tn + fn, basis="L1 = 0 sample"),
    dict(measure="sensitivity (derived)", estimate=sens, ci_low=np.nan, ci_high=np.nan, n=np.nan,
         basis=f"re-weighted with L1 prevalence {prev:.3f}"),
    dict(measure="specificity (derived)", estimate=spec, ci_low=np.nan, ci_high=np.nan, n=np.nan,
         basis=f"re-weighted with L1 prevalence {prev:.3f}"),
])
save_table(out, "e7_registrar_check", args.mode)
say(f"Next: python 04_src/14_tables_figures.py --mode {args.mode}")
