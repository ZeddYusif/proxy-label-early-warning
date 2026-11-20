"""Step 01. Make synthetic practice data in the same file formats as the real data.

Command:  python 04_src/01_make_demo_data.py
Writes:   02_data/demo/oulad/*.csv   (OULAD-like: courses, studentInfo, studentRegistration,
                                       studentVle, assessments, studentAssessment)
          02_data/demo/knust/*.csv   (KNUST-like: activity_log, gradebook, quiz_attempts,
                                       plus a practice Registrar 2 by 2 table)
Next:     python 04_src/03_adapt_oulad.py --mode demo

These files are invented. They exist only so you can rehearse every step safely.
A hidden truth drives them: some students truly withdraw, and some only stop for a while
(for example poor connectivity on mobile). That gap between truth and proxy is what the
study measures. Never report any number made from these files.
"""
import numpy as np
import pandas as pd

from common import ROOT, load_config, say

rng = np.random.default_rng(42)
cfg = load_config()
out_o = ROOT / cfg["paths"]["demo_raw"] / "oulad"
out_k = ROOT / cfg["paths"]["demo_raw"] / "knust"
out_o.mkdir(parents=True, exist_ok=True)
out_k.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------------ OULAD-like
modules = ["AAA", "BBB"]
presentations = {"2013B": 240, "2013J": 262, "2014B": 241, "2014J": 269}
per_course = 250
courses, info, reg, vle, assess, sassess = [], [], [], [], [], []
sid = 10000
aid = 1000
for mod in modules:
    for pres, length in presentations.items():
        courses.append((mod, pres, length))
        dues = [int(length * f) for f in (0.10, 0.30, 0.50, 0.70, 0.90)]
        a_ids = []
        for d in dues:
            aid += 1
            a_ids.append((aid, d))
            assess.append((mod, pres, aid, "TMA", d, 20.0))
        aid += 1
        assess.append((mod, pres, aid, "Exam", length, 100.0))
        drift = 0.25 if pres.startswith("2014") else 0.0          # later years withdraw a little more
        for _ in range(per_course):
            sid += 1
            e = rng.normal()
            withdraw = rng.random() < sigmoid(-1.3 - 0.9 * e + drift)
            unreg = None
            if withdraw:
                unreg = int(rng.uniform(-8, length * 0.8))
                stop_day = max(-14, unreg - int(rng.uniform(0, 15)))
            else:
                stop_day = length
            late_fade = (not withdraw) and rng.random() < 0.15        # finishes quietly, not withdrawn
            if late_fade:
                stop_day = length - int(rng.uniform(22, 40))
            gap = None
            if (not withdraw) and rng.random() < 0.20:               # a temporary stop-out
                g0 = int(rng.uniform(20, length - 60))
                gap = (g0, g0 + int(rng.uniform(15, 30)))
            p_active = sigmoid(-0.4 + 0.8 * e)
            for day in range(-14, min(stop_day, length) + 1):
                if gap and gap[0] <= day < gap[1]:
                    continue
                if rng.random() < p_active:
                    vle.append((mod, pres, sid, 500 + day % 7, day, int(rng.poisson(2 + 2 * np.exp(0.5 * e))) + 1))
            for a, d in a_ids:
                if d > stop_day or (gap and gap[0] <= d < gap[1]):
                    continue
                if rng.random() < sigmoid(2.2 + 0.8 * e):
                    sub = d - int(rng.uniform(0, 5)) if rng.random() > 0.06 else d + int(rng.uniform(1, 9))
                    sassess.append((a, sid, sub, 0, float(np.clip(rng.normal(68 + 6 * e, 12), 0, 100))))
            if not withdraw and not late_fade and rng.random() < 0.7:
                sassess.append((aid, sid, length, 0, float(np.clip(rng.normal(60 + 6 * e, 15), 0, 100))))
            result = "Withdrawn" if withdraw else rng.choice(["Pass", "Fail", "Distinction"], p=[0.6, 0.3, 0.1])
            info.append((mod, pres, sid, rng.choice(["M", "F"]), result))
            reg.append((mod, pres, sid, -int(rng.uniform(10, 60)), unreg))

pd.DataFrame(courses, columns=["code_module", "code_presentation", "module_presentation_length"]).to_csv(out_o / "courses.csv", index=False)
pd.DataFrame(info, columns=["code_module", "code_presentation", "id_student", "gender", "final_result"]).to_csv(out_o / "studentInfo.csv", index=False)
pd.DataFrame(reg, columns=["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"]).to_csv(out_o / "studentRegistration.csv", index=False)
pd.DataFrame(vle, columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]).to_csv(out_o / "studentVle.csv", index=False)
pd.DataFrame(assess, columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"]).to_csv(out_o / "assessments.csv", index=False)
pd.DataFrame(sassess, columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]).to_csv(out_o / "studentAssessment.csv", index=False)
say(f"OULAD-like demo: {len(info)} registrations, {len(vle):,} daily click rows -> {out_o.relative_to(ROOT)}")

# ------------------------------------------------------------------ KNUST-like
kc = cfg["knust"]
start = pd.Timestamp(kc["course_start"])
end = pd.Timestamp(kc["export_end"])
length = (end - start).days
quiz_days = [20, 45, 70, 95, 120, 140]
n_k = 1200
log, grades, attempts, truth = [], [], [], []
types = ["course_viewed", "resource_viewed", "quiz_viewed", "forum_viewed", "assignment_viewed"]
for i in range(1, n_k + 1):
    pid = f"P{i:05d}"
    e = rng.normal()
    mobile = rng.random() < 0.55
    withdraw = rng.random() < sigmoid(-2.0 - 0.9 * e)
    stop_day = int(rng.uniform(5, length * 0.7)) if withdraw else length
    gap = None
    if not withdraw and rng.random() < (0.35 if mobile else 0.12):     # connectivity stop-outs, more on mobile
        g0 = int(rng.uniform(10, length - 10))
        gap = (g0, g0 + int(rng.uniform(18, 32)))
    p_active = sigmoid(-0.6 + 0.8 * e)
    first = 0 if rng.random() > 0.35 else int(rng.uniform(5, 19))      # late starters, common in the real export
    for day in range(-10 if first == 0 else first, stop_day + 1):
        if gap and gap[0] <= day < gap[1]:
            continue
        if rng.random() < p_active:
            for _ in range(int(rng.poisson(3)) + 1):
                t = start + pd.Timedelta(days=day, seconds=int(rng.uniform(6 * 3600, 23 * 3600)))
                src = ("mobile_app" if rng.random() < 0.85 else "web") if mobile else ("web" if rng.random() < 0.85 else "mobile_app")
                log.append((pid, t.strftime("%Y-%m-%d %H:%M:%S"), rng.choice(types), src))
    row = [pid]
    for q, qd in enumerate(quiz_days, start=1):
        done = first <= qd <= stop_day and not (gap and gap[0] <= qd < gap[1]) and rng.random() < sigmoid(3.4 + 0.7 * e)
        if done:
            score = float(np.clip(rng.normal(6.5 + e, 1.8), 0, 10))
            t = start + pd.Timedelta(days=qd - int(rng.uniform(0, 4)), hours=int(rng.uniform(8, 22)))
            attempts.append((pid, f"quiz{q}", "finished", round(score, 2), t.strftime("%Y-%m-%d %H:%M:%S")))
            row.append(round(score, 2))
        else:
            if rng.random() < 0.1:
                attempts.append((pid, f"quiz{q}", "inprogress", None, None))
            row.append(None)
    grades.append(row)
    truth.append((pid, int(withdraw)))
# a few rows from an earlier year, and exact duplicates, as in the real export
log += [(f"P{i:05d}", "2025-09-15 10:00:00", "course_viewed", "web") for i in range(1, 40)]
log_df = pd.DataFrame(log, columns=[kc["activity_id_col"], kc["activity_time_col"], kc["activity_type_col"], kc["activity_source_col"]])
log_df = pd.concat([log_df, log_df.sample(frac=0.01, random_state=1)], ignore_index=True)
log_df.to_csv(out_k / kc["activity_file"], index=False)
pd.DataFrame(grades, columns=[kc["gradebook_id_col"]] + kc["gradebook_quiz_cols"]).to_csv(out_k / kc["gradebook_file"], index=False)
pd.DataFrame(attempts, columns=[kc["attempts_id_col"], kc["attempts_quiz_col"], kc["attempts_state_col"],
                                kc["attempts_score_col"], kc["attempts_finish_col"]]).to_csv(out_k / kc["attempts_file"], index=False)
pd.DataFrame(truth, columns=["participant_id", "truly_withdrawn"]).to_csv(out_k / "hidden_truth_DEMO_ONLY.csv", index=False)
say(f"KNUST-like demo: {n_k} participants, {len(log_df):,} log rows -> {out_k.relative_to(ROOT)}")
say("Next: python 04_src/03_adapt_oulad.py --mode demo")
