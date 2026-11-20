"""Label rules L1 to L5 (and L0 on OULAD). Pure functions, so tests can check them.

Inputs are the standard tables written by the adapters:
  students:    student_key, course, presentation, length_days, official_withdrawn, unreg_day, channel
  events:      student_key, day, n_events                       (one row per active day)
  assessments: student_key, order, due_day, has_result, result_day, score

The label builder never reads a feature column. Features never read a label column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def missed_flags(assessments: pd.DataFrame, version: str) -> pd.Series:
    """True when an assessment counts as missed.

    emulated: no result at all (the only rule the KNUST export allows).
    due:      no result by the due date (OULAD has due dates).
    """
    no_result = ~assessments["has_result"].astype(bool)
    if version == "emulated":
        return no_result
    late = assessments["due_day"].notna() & assessments["result_day"].notna() & (
        assessments["result_day"] > assessments["due_day"])
    return no_result | late


def build_labels(students: pd.DataFrame, events: pd.DataFrame, assessments: pd.DataFrame,
                 version: str, inactivity_days: int, consecutive: int, last_quarter: float,
                 pre_start_days: int) -> pd.DataFrame:
    """Return one row per student with L1..L5 and the day each label's signal began (onset)."""
    s = students[["student_key", "length_days"]].copy()

    ev = events.merge(s, on="student_key", how="inner")
    ev = ev[(ev["n_events"] > 0) & (ev["day"] <= ev["length_days"]) & (ev["day"] >= -pre_start_days)]
    last = ev.groupby("student_key")["day"].max().rename("last_active")
    s = s.merge(last, on="student_key", how="left")
    s["last_active"] = s["last_active"].fillna(-pre_start_days - 1)
    s["inactive_run"] = s["length_days"] - s["last_active"]
    s["onset_inactive"] = s["last_active"] + 1

    # L4: inactivity only
    s["L4"] = (s["inactive_run"] >= inactivity_days).astype(int)

    # L3: Moodle default, no activity in the last quarter of the course
    s["L3"] = (s["last_active"] < np.ceil(last_quarter * s["length_days"])).astype(int)

    # assessments: missed flags in module order
    a = assessments.sort_values(["student_key", "order"]).copy()
    a["missed"] = missed_flags(a, version).astype(int)
    l2 = a.groupby("student_key")["missed"].max().rename("L2")
    # first due day of a missed assessment (only meaningful with due dates)
    first_missed_due = (a[a["missed"] == 1].groupby("student_key")["due_day"].min().rename("onset_L2"))

    a["next_missed"] = a.groupby("student_key")["missed"].shift(-1)
    pair = a[(a["missed"] == 1) & (a["next_missed"] == 1)] if consecutive == 2 else None
    if consecutive != 2:
        raise ValueError("Only two consecutive missed assessments is implemented. Changing it changes a label rule after the lock: record it as a deviation.")
    l5 = pair.groupby("student_key").size().gt(0).astype(int).rename("L5")
    onset_l5 = pair.groupby("student_key")["due_day"].min().rename("onset_L5")

    s = (s.merge(l2, on="student_key", how="left").merge(first_missed_due, on="student_key", how="left")
          .merge(l5, on="student_key", how="left").merge(onset_l5, on="student_key", how="left"))
    s["L2"] = s["L2"].fillna(0).astype(int)
    s["L5"] = s["L5"].fillna(0).astype(int)
    if version == "emulated":                       # no due dates: onset cannot be known
        s["onset_L2"] = np.nan
        s["onset_L5"] = np.nan

    s["L1"] = ((s["L4"] == 1) & (s["L5"] == 1)).astype(int)
    s["onset_L1"] = np.where(s["onset_L5"].notna(), np.maximum(s["onset_inactive"], s["onset_L5"]), s["onset_inactive"])
    s["onset_L3"] = s["onset_inactive"]
    s["onset_L4"] = s["onset_inactive"]
    keep = ["student_key"] + [f"L{i}" for i in range(1, 6)] + [f"onset_L{i}" for i in range(1, 6)]
    return s[keep]


def sensitivity_labels(students, events, assessments, version, thresholds, consecutive, last_quarter, pre):
    """L1 and L4 again with other inactivity thresholds (sensitivity check, not new labels)."""
    out = students[["student_key"]].copy()
    for t in thresholds:
        lab = build_labels(students, events, assessments, version, t, consecutive, last_quarter, pre)
        out = out.merge(lab[["student_key", "L1", "L4"]].rename(columns={"L1": f"L1_inact{t}", "L4": f"L4_inact{t}"}),
                        on="student_key")
    return out
