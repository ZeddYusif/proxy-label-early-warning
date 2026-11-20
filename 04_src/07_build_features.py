"""Step 07. Build the shared features for each prediction window (14, 28 and 42 days).

Command:  python 04_src/07_build_features.py --mode demo   (or --mode real)
Writes:   02_data/processed/<mode>/<dataset>/features_T<window>.csv
          06_results/<mode>/tables/feature_parity.csv   (real mode with features.source: external)
Next:     python 04_src/08_e2_proxy_validity.py --mode <same mode>

If your frozen pipeline already writes these features, keep it as the source of truth:
set features.source: external in config.yaml and point external_pattern at your files.
This script then checks that its own recipe agrees with yours (parity table) and the
experiments read your files. A large disagreement means the two recipes differ. Find the
feature that differs and correct the recipe before you go on (Workflow Guide, section 3.3).
"""
import numpy as np
import pandas as pd

from common import ROOT, load_config, parse_mode, processed_dir, require_lock, save_table, say
from features_lib import build_features

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
parity = []
for ds in ["oulad", "knust"]:
    d = processed_dir(args.mode, ds)
    st = pd.read_csv(d / "students.csv")
    ev = pd.read_csv(d / "events.csv")
    asmt = pd.read_csv(d / "assessments.csv")
    for T in cfg["windows"]:
        f = build_features(st, ev, asmt, T, cfg["pre_start_days"])
        f.to_csv(d / f"features_T{T}.csv", index=False)
        say(f"  {ds} T={T}: {len(f):,} rows, no activity before T: {(f['active_ratio'] == 0).mean():.0%}")
        ext = ROOT / cfg["features"]["external_pattern"].format(dataset=ds, window=T)
        if args.mode == "real" and cfg["features"]["source"] == "external" and ext.exists():
            e = pd.read_csv(ext)
            m = f.merge(e, on="student_key", suffixes=("_starter", "_yours"))
            for c in cfg["features"]["shared"]:
                if f"{c}_yours" in m:
                    a, b = m[f"{c}_starter"], m[f"{c}_yours"]
                    ok = a.notna() & b.notna()
                    parity.append(dict(dataset=ds, window=T, feature=c, n=int(ok.sum()),
                                       correlation=float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                                       max_abs_diff=float((a[ok] - b[ok]).abs().max()) if ok.any() else np.nan))
if parity:
    save_table(pd.DataFrame(parity), "feature_parity", args.mode)
say(f"Next: python 04_src/08_e2_proxy_validity.py --mode {args.mode}")
