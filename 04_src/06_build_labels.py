"""Step 06. Build the labels: L0 (OULAD official), L1 to L5 (proxies), and the onset day of each.

Command:  python 04_src/06_build_labels.py --mode demo   (or --mode real, after the lock)
Writes:   02_data/processed/<mode>/oulad/labels.csv   (L1..L5 in two versions: _due and _emulated)
          02_data/processed/<mode>/knust/labels.csv   (L1..L5, KNUST has no due dates)
          06_results/<mode>/tables/label_prevalence.csv
Next:     python 04_src/07_build_features.py --mode <same mode>

Label codes (Workflow Guide, section 4.1):
  L0 official withdrawal (OULAD only)        L1 21+ inactive days to the end AND two consecutive missed
  L2 any assessment without a result         L3 Moodle default: no activity in the last quarter
  L4 21+ inactive days to the end            L5 two consecutive assessments without a result
"""
import pandas as pd

from common import load_config, parse_mode, processed_dir, require_lock, save_table, say, stop
from labels_lib import build_labels, sensitivity_labels

args = parse_mode(__doc__)
require_lock(args.mode)
cfg = load_config()
lc = cfg["labels"]
pre = cfg["pre_start_days"]
rows = []
for ds in ["oulad", "knust"]:
    d = processed_dir(args.mode, ds)
    if not (d / "students.csv").exists():
        stop(f"{d}/students.csv missing. Run the adapter for {ds} first (step 03 or 04).")
    st = pd.read_csv(d / "students.csv")
    ev = pd.read_csv(d / "events.csv")
    asmt = pd.read_csv(d / "assessments.csv")
    out = st[["student_key", "course", "presentation", "unreg_day", "channel"]].copy()
    if ds == "oulad":
        out["L0"] = st["official_withdrawn"].astype(int)
        out["onset_L0"] = st["unreg_day"]
        versions = lc["oulad_versions"]
    else:
        versions = ["emulated"]
    for v in versions:
        lab = build_labels(st, ev, asmt, v, lc["inactivity_days"], lc["consecutive_missed"],
                           lc["last_quarter_start"], pre)
        sens = sensitivity_labels(st, ev, asmt, v, lc["inactivity_sensitivity"], lc["consecutive_missed"],
                                  lc["last_quarter_start"], pre)
        lab = lab.merge(sens, on="student_key")
        suffix = f"_{v}" if ds == "oulad" else ""
        lab = lab.rename(columns={c: c + suffix for c in lab.columns if c != "student_key"})
        out = out.merge(lab, on="student_key", how="left")
    out.to_csv(d / "labels.csv", index=False)
    lab_cols = [c for c in out.columns if c.startswith("L")]
    for c in lab_cols:
        rows.append(dict(dataset=ds, label=c, n=len(out), positives=int(out[c].sum()), prevalence=out[c].mean()))
    say(f"  {ds}: {len(out):,} students, labels: {', '.join(lab_cols)}")
save_table(pd.DataFrame(rows), "label_prevalence", args.mode)
say(f"Next: python 04_src/07_build_features.py --mode {args.mode}")
