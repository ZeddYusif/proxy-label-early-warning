"""Step 04. Turn the KNUST Moodle exports into the three standard tables.

Command:  python 04_src/04_adapt_knust.py --mode demo     (practice files)
          python 04_src/04_adapt_knust.py --mode real     (your three KNUST exports, on your laptop only)
Writes:   02_data/processed/<mode>/knust/{students,events,assessments}.csv
Next:     python 04_src/05_lock_plan.py   (real mode)   or   python 04_src/06_build_labels.py --mode demo

Before a real run, set every column name in the knust section of config.yaml (lines marked CHECK).
Rules applied here, as in your draft:
  exact duplicate log rows are removed; one log row counts as one event;
  days are counted from the course start; events from up to 14 days before the start are kept;
  a quiz counts as having a result when the gradebook holds a grade;
  its result day is the finish time of the first finished attempt (if no time exists, the day is unknown).
The export has no due dates and no withdrawal record, so official_withdrawn and unreg_day stay empty.
"""
import pandas as pd

from common import ROOT, load_config, parse_mode, processed_dir, say, stop

args = parse_mode(__doc__)
cfg = load_config()
kc = cfg["knust"]
src = ROOT / (cfg["paths"]["demo_raw"] + "/knust" if args.mode == "demo" else cfg["paths"]["knust_raw"])
for f in [kc["activity_file"], kc["gradebook_file"], kc["attempts_file"]]:
    if not (src / f).exists():
        stop(f"{f} not found in {src.relative_to(ROOT)}. Check the file names in config.yaml (knust section).")
out = processed_dir(args.mode, "knust")
start = pd.Timestamp(kc["course_start"])
length = (pd.Timestamp(kc["export_end"]) - start).days
pre = cfg["pre_start_days"]

log = pd.read_csv(src / kc["activity_file"])
need = [kc["activity_id_col"], kc["activity_time_col"], kc["activity_source_col"]]
missing = [c for c in need if c not in log.columns]
if missing:
    stop(f"activity file has no column(s) {missing}. Its columns are {list(log.columns)}. Edit config.yaml.")
n0 = len(log)
log = log.drop_duplicates()
say(f"  activity log: {n0:,} rows, {n0 - len(log):,} exact duplicates removed")
log["t"] = pd.to_datetime(log[kc["activity_time_col"]], format=kc.get("time_format"), errors="coerce", utc=True).dt.tz_localize(None)
bad = log["t"].isna().sum()
if bad:
    say(f"  WARNING: {bad:,} event times could not be read and are left out. Check time_format in config.yaml.")
all_ids = set(log[kc["activity_id_col"]].astype(str))
log["day"] = (log["t"].dt.normalize() - start).dt.days
log = log[(log["day"] >= -pre) & (log["day"] <= length)]
log["student_key"] = log[kc["activity_id_col"]].astype(str)
events = log.groupby(["student_key", "day"], as_index=False).size().rename(columns={"size": "n_events"})
channel = (log.groupby("student_key")[kc["activity_source_col"]]
           .agg(lambda s: s.value_counts().idxmax()).rename("channel"))

gb = pd.read_csv(src / kc["gradebook_file"])
if "grade_item" in gb.columns and kc["gradebook_quiz_cols"][0] not in gb.columns:
    gb = gb.dropna(subset=["grade_item"])
    gb = gb.pivot_table(index=kc["gradebook_id_col"], columns="grade_item", values="finalgrade", aggfunc="first").reset_index()
gb["student_key"] = gb[kc["gradebook_id_col"]].astype(str)
quiz_cols = kc["gradebook_quiz_cols"]
missing = [c for c in quiz_cols if c not in gb.columns]
if missing:
    stop(f"gradebook has no column(s) {missing}. Its columns are {list(gb.columns)}. Edit config.yaml.")

ids = sorted(set(events["student_key"]) | set(gb["student_key"]) | all_ids)
students = pd.DataFrame({"student_key": ids, "course": "KNUST_2026", "presentation": "2026",
                         "length_days": length, "official_withdrawn": pd.NA, "unreg_day": pd.NA})
students = students.merge(channel, on="student_key", how="left")
students.to_csv(out / "students.csv", index=False)
events.to_csv(out / "events.csv", index=False)

att = pd.read_csv(src / kc["attempts_file"])
att["student_key"] = att[kc["attempts_id_col"]].astype(str)
att = att[att[kc["attempts_state_col"]] == kc["attempts_finished_value"]].copy()
att["t"] = pd.to_datetime(att[kc["attempts_finish_col"]], format=kc.get("time_format"), errors="coerce", utc=True).dt.tz_localize(None)
att["result_day"] = (att["t"].dt.normalize() - start).dt.days
first_fin = att.groupby(["student_key", kc["attempts_quiz_col"]])["result_day"].min()

rows = []
for order, q in enumerate(quiz_cols, start=1):
    sub = gb[["student_key", q]].rename(columns={q: "score"})
    sub["assessment_id"] = q
    sub["order"] = order
    rows.append(sub)
assessments = pd.concat(rows, ignore_index=True)
assessments = assessments[assessments["student_key"].isin(ids)]
assessments["score"] = pd.to_numeric(assessments["score"], errors="coerce")
assessments["has_result"] = assessments["score"].notna().astype(int)
assessments["due_day"] = pd.NA
idx = pd.MultiIndex.from_arrays([assessments["student_key"], assessments["assessment_id"]])
assessments["result_day"] = first_fin.reindex(idx).to_numpy()
assessments.loc[assessments["has_result"] == 0, "result_day"] = pd.NA
assessments[["student_key", "assessment_id", "order", "due_day", "has_result", "result_day", "score"]].to_csv(
    out / "assessments.csv", index=False)
no_time = int(((assessments["has_result"] == 1) & assessments["result_day"].isna()).sum())
say(f"  students: {len(students):,}  events: {len(events):,} student-days  quiz results: {int(assessments['has_result'].sum()):,}")
if no_time:
    say(f"  NOTE: {no_time:,} graded quizzes have no finish time. They count for labels but not as early features.")
say(f"  main access channel: {students['channel'].value_counts(dropna=False).to_dict()}")
nxt = "python 04_src/05_lock_plan.py" if args.mode == "real" else "python 04_src/06_build_labels.py --mode demo"
say(f"Wrote {out.relative_to(ROOT)}. Next: {nxt}")
