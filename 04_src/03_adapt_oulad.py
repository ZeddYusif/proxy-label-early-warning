"""Step 03. Turn the OULAD files into the three standard tables.

Command:  python 04_src/03_adapt_oulad.py --mode demo     (practice files)
          python 04_src/03_adapt_oulad.py --mode real     (the real OULAD download)
Writes:   02_data/processed/<mode>/oulad/{students,events,assessments}.csv
Next:     python 04_src/04_adapt_knust.py --mode <same mode>

Student key = module + presentation + student id, because one student can register twice.
Exam rows are left out of the assessment rules (they are sparse in studentAssessment).
studentVle is read in chunks, so a laptop with 8 GB of memory copes with the real file.
"""
import pandas as pd

from common import ROOT, load_config, parse_mode, processed_dir, say, stop

args = parse_mode(__doc__)
cfg = load_config()
src = ROOT / (cfg["paths"]["demo_raw"] + "/oulad" if args.mode == "demo" else cfg["paths"]["oulad_raw"])
if not (src / "studentInfo.csv").exists():
    stop(f"no OULAD files in {src.relative_to(ROOT)}. "
         + ("Run 04_src/01_make_demo_data.py first." if args.mode == "demo" else "See the Workflow Guide, section 3.1."))
out = processed_dir(args.mode, "oulad")


def key(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)


courses = pd.read_csv(src / "courses.csv")
info = pd.read_csv(src / "studentInfo.csv")
reg = pd.read_csv(src / "studentRegistration.csv")
st = info.merge(reg, on=["code_module", "code_presentation", "id_student"], how="left").merge(
    courses, on=["code_module", "code_presentation"], how="left")
students = pd.DataFrame({
    "student_key": key(st),
    "course": st["code_module"] + "_" + st["code_presentation"],
    "presentation": st["code_presentation"],
    "length_days": st["module_presentation_length"],
    "official_withdrawn": (st["final_result"] == "Withdrawn").astype(int),
    "unreg_day": pd.to_numeric(st["date_unregistration"], errors="coerce"),
    "channel": pd.NA,
})
students.to_csv(out / "students.csv", index=False)
say(f"  students: {len(students):,} registrations, {students['official_withdrawn'].mean():.1%} withdrawn")

parts = []
for chunk in pd.read_csv(src / "studentVle.csv", chunksize=2_000_000,
                         usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"]):
    chunk["student_key"] = key(chunk)
    parts.append(chunk.groupby(["student_key", "date"], as_index=False)["sum_click"].sum())
events = pd.concat(parts).groupby(["student_key", "date"], as_index=False)["sum_click"].sum()
events = events.rename(columns={"date": "day", "sum_click": "n_events"})
events.to_csv(out / "events.csv", index=False)
say(f"  events: {len(events):,} student-days")

ass = pd.read_csv(src / "assessments.csv")
ass = ass[~ass["assessment_type"].isin(cfg["labels"]["oulad_exclude_assessment_types"])]
ass["course"] = ass["code_module"] + "_" + ass["code_presentation"]
ass = ass.sort_values(["course", "date", "id_assessment"])
ass["order"] = ass.groupby("course").cumcount() + 1
grid = students[["student_key", "course"]].merge(ass[["course", "id_assessment", "order", "date"]], on="course")
sub = pd.read_csv(src / "studentAssessment.csv")
grid = grid.merge(st.assign(student_key=key(st))[["student_key", "id_student"]], on="student_key")
grid = grid.merge(sub[["id_assessment", "id_student", "date_submitted", "score"]],
                  on=["id_assessment", "id_student"], how="left")
assessments = pd.DataFrame({
    "student_key": grid["student_key"],
    "assessment_id": grid["id_assessment"],
    "order": grid["order"],
    "due_day": pd.to_numeric(grid["date"], errors="coerce"),
    "has_result": grid["date_submitted"].notna().astype(int),
    "result_day": pd.to_numeric(grid["date_submitted"], errors="coerce"),
    "score": pd.to_numeric(grid["score"], errors="coerce"),
})
assessments.to_csv(out / "assessments.csv", index=False)
say(f"  assessments: {len(assessments):,} student-assessment rows ({ass['id_assessment'].nunique()} assessments)")
say(f"Wrote {out.relative_to(ROOT)}. Next: python 04_src/04_adapt_knust.py --mode {args.mode}")
