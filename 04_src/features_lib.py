"""The shared, scale-free feature recipe (Table M1 of the draft), built from the standard tables.

If your frozen pipeline builds the features, it stays the source of truth (set
features.source: external in config.yaml). This recipe is used for the demo and as a parity check.
Only events dated before the prediction day T, and results dated before T, are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_COLS = ["clicks_total", "active_days", "clicks_per_active_day", "clicks_last_week", "click_trend", "n_early_assess"]


def build_features(students: pd.DataFrame, events: pd.DataFrame, assessments: pd.DataFrame,
                   window: int, pre_start_days: int) -> pd.DataFrame:
    T = window
    wlen = T + pre_start_days
    base = students[["student_key", "course", "presentation"]].copy()

    ev = events[(events["day"] < T) & (events["day"] >= -pre_start_days) & (events["n_events"] > 0)]
    g = ev.groupby("student_key")
    f = pd.DataFrame({
        "clicks_total": g["n_events"].sum(),
        "active_days": g["day"].nunique(),
        "last_day": g["day"].max(),
        "first_day": g["day"].min(),
        "clicks_last_week": ev[ev["day"] >= T - 7].groupby("student_key")["n_events"].sum(),
        "clicks_first_week": ev[(ev["day"] >= 0) & (ev["day"] < 7)].groupby("student_key")["n_events"].sum(),
    })
    df = base.merge(f, left_on="student_key", right_index=True, how="left")
    for c in ["clicks_total", "active_days", "clicks_last_week", "clicks_first_week"]:
        df[c] = df[c].fillna(0)
    df["active_ratio"] = df["active_days"] / wlen
    df["clicks_per_active_day"] = np.where(df["active_days"] > 0, df["clicks_total"] / df["active_days"].clip(lower=1), 0.0)
    df["gap_frac"] = np.where(df["last_day"].notna(), (T - df["last_day"]) / wlen, 1.0)
    df["first_click_frac"] = np.where(df["first_day"].notna(), (df["first_day"] + pre_start_days) / wlen, 1.0)
    df["click_trend"] = df["clicks_last_week"] - df["clicks_first_week"]
    df["active_last7"] = (df["clicks_last_week"] > 0).astype(int)      # used by E4, never a feature

    a = assessments[assessments["has_result"].astype(bool) & assessments["result_day"].notna() & (assessments["result_day"] < T)]
    ga = a.groupby("student_key")
    fa = pd.DataFrame({"n_early_assess": ga.size(), "early_score_mean": ga["score"].mean()})
    df = df.merge(fa, left_on="student_key", right_index=True, how="left")
    df["n_early_assess"] = df["n_early_assess"].fillna(0)
    df["has_early_assess"] = (df["n_early_assess"] > 0).astype(int)

    # z-scores within each course, computed without labels
    for c in Z_COLS:
        grp = df.groupby("course")[c]
        sd = grp.transform("std").replace(0, np.nan)
        df[f"{c}_z"] = ((df[c] - grp.transform("mean")) / sd).fillna(0.0)

    keep = ["student_key", "course", "presentation", "clicks_total_z", "active_days_z", "active_ratio",
            "clicks_per_active_day_z", "gap_frac", "first_click_frac", "clicks_last_week_z", "click_trend_z",
            "n_early_assess_z", "has_early_assess", "early_score_mean", "active_last7"]
    return df[keep]
