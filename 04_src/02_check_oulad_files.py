"""Step 02 (real data only). Check the seven OULAD files and record them in the datasheet.

Command:  python 04_src/02_check_oulad_files.py
Before:   download OULAD from the Open University page (Workflow Guide, section 3.1) and put the
          seven CSV files, unchanged, in 02_data/raw/oulad/. The files are never committed.
Writes:   a dated entry in 02_data/datasheet.md with row counts and SHA-256 fingerprints.
Next:     python 04_src/03_adapt_oulad.py --mode real
"""
import datetime as dt
import hashlib

from common import ROOT, load_config, say, stop

FILES = ["courses.csv", "assessments.csv", "vle.csv", "studentInfo.csv",
         "studentRegistration.csv", "studentAssessment.csv", "studentVle.csv"]
cfg = load_config()
folder = ROOT / cfg["paths"]["oulad_raw"]
missing = [f for f in FILES if not (folder / f).exists()]
if missing:
    stop(f"missing in {folder.relative_to(ROOT)}: {', '.join(missing)}. See the Workflow Guide, section 3.1.")

lines = [f"\n## OULAD files checked {dt.date.today().isoformat()}\n", "| File | Rows | SHA-256 |", "|---|---|---|"]
for f in FILES:
    p = folder / f
    h = hashlib.sha256()
    rows = -1
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            rows += chunk.count(b"\n")
    lines.append(f"| {f} | {rows:,} | {h.hexdigest()} |")
    say(f"  {f:<24} {rows:>12,} rows")
ds = ROOT / "02_data" / "datasheet.md"
with open(ds, "a", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
say(f"Recorded in {ds.relative_to(ROOT)}. Compare the counts with your draft (32,593 registrations).")
say("Next: python 04_src/03_adapt_oulad.py --mode real")
