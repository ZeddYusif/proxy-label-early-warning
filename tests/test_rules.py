"""Checks of every rule the study depends on. Run:  python -m pytest -q

If a test fails after you edit code, the edit broke a rule. Undo it with git restore, then find the cause (Workflow Guide, section S1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_src"))
from common import jaccard, top_mask  # noqa: E402
from features_lib import build_features  # noqa: E402
from labels_lib import build_labels  # noqa: E402
from stats_tools import corrected_compare, kappa_from_counts  # noqa: E402

LENGTH = 100


def _students(keys):
    return pd.DataFrame({"student_key": keys, "course": "C1", "presentation": "P", "length_days": LENGTH})


def _events(spec):
    rows = [(k, d, 3) for k, days in spec.items() for d in days]
    return pd.DataFrame(rows, columns=["student_key", "day", "n_events"])


def _assess(spec, dues=(10, 30, 50, 70)):
    rows = []
    for k, results in spec.items():
        for i, (due, res) in enumerate(zip(dues, results), start=1):
            rows.append((k, i, due, int(res is not None), res, 60.0 if res is not None else np.nan))
    return pd.DataFrame(rows, columns=["student_key", "order", "due_day", "has_result", "result_day", "score"])


def _labels(ev, asmt, keys, version="due"):
    return build_labels(_students(keys), ev, asmt, version, 21, 2, 0.75, 14).set_index("student_key")


def test_active_completer_has_no_labels():
    ev = _events({"a": range(0, 101, 2)})
    asmt = _assess({"a": [9, 29, 49, 69]})
    lab = _labels(ev, asmt, ["a"]).loc["a"]
    assert (lab[["L1", "L2", "L3", "L4", "L5"]] == 0).all()


def test_stopper_is_L1_L3_L4_L5():
    ev = _events({"b": range(0, 41)})                 # stops after day 40
    asmt = _assess({"b": [9, 29, None, None]})        # misses the last two
    lab = _labels(ev, asmt, ["b"]).loc["b"]
    assert lab["L1"] == 1 and lab["L3"] == 1 and lab["L4"] == 1 and lab["L5"] == 1 and lab["L2"] == 1
    assert lab["onset_L4"] == 41 and lab["onset_L5"] == 50 and lab["onset_L1"] == 50


def test_short_gap_at_end_is_not_L4():
    ev = _events({"c": list(range(0, 85))})           # inactive for the last 15 days only
    asmt = _assess({"c": [9, 29, 49, 69]})
    lab = _labels(ev, asmt, ["c"]).loc["c"]
    assert lab["L4"] == 0 and lab["L1"] == 0 and lab["L3"] == 0


def test_one_missed_quiz_is_L2_not_L5():
    ev = _events({"d": range(0, 101)})
    asmt = _assess({"d": [9, None, 49, 69]})
    lab = _labels(ev, asmt, ["d"]).loc["d"]
    assert lab["L2"] == 1 and lab["L5"] == 0 and lab["L1"] == 0


def test_late_submission_counts_as_missed_only_with_due_dates():
    ev = _events({"e": range(0, 101)})
    asmt = _assess({"e": [9, 35, 55, 69]})            # quizzes 2 and 3 handed in after their due days
    assert _labels(ev, asmt, ["e"], "due").loc["e", "L5"] == 1
    assert _labels(ev, asmt, ["e"], "emulated").loc["e", "L5"] == 0


def test_emulated_version_has_no_assessment_onset():
    ev = _events({"f": range(0, 30)})
    asmt = _assess({"f": [9, None, None, None]})
    lab = _labels(ev, asmt, ["f"], "emulated").loc["f"]
    assert np.isnan(lab["onset_L5"]) and lab["L5"] == 1


def test_features_ignore_events_after_prediction_day():
    st = _students(["g"])
    asmt = _assess({"g": [9, 29, 49, 69]})
    ev1 = _events({"g": range(0, 28)})
    ev2 = _events({"g": list(range(0, 28)) + list(range(28, 100))})
    f1 = build_features(st, ev1, asmt, 28, 14).drop(columns=["student_key", "course", "presentation"])
    f2 = build_features(st, ev2, asmt, 28, 14).drop(columns=["student_key", "course", "presentation"])
    pd.testing.assert_frame_equal(f1, f2)


def test_features_ignore_results_after_prediction_day():
    st = _students(["h"])
    ev = _events({"h": range(0, 28)})
    f1 = build_features(st, ev, _assess({"h": [9, None, None, None]}), 28, 14)
    f2 = build_features(st, ev, _assess({"h": [9, 29, 49, 69]}), 28, 14)
    assert f1["early_score_mean"].iloc[0] == f2["early_score_mean"].iloc[0]


def test_feature_table_holds_no_label_column():
    st = _students(["i"])
    f = build_features(st, _events({"i": range(0, 10)}), _assess({"i": [9, 29, 49, 69]}), 28, 14)
    assert not any(c.startswith("L") and c[1:2].isdigit() for c in f.columns)


def test_top_mask_flags_exact_share_and_is_repeatable():
    s = np.r_[np.ones(50), np.zeros(50)]
    m1, m2 = top_mask(s, 0.2, 3), top_mask(s, 0.2, 3)
    assert m1.sum() == 20 and (m1 == m2).all() and s[m1].min() == 1


def test_jaccard():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert jaccard(a, b) == 1 / 3


def test_kappa_perfect_and_chance():
    assert kappa_from_counts(50, 0, 0, 50) == 1.0
    assert abs(kappa_from_counts(25, 25, 25, 25)) < 1e-12


def test_equivalence_rule_three_words():
    rng = np.random.default_rng(0)
    same = rng.normal(0.0, 0.005, 50)
    assert corrected_compare(same, 800, 200, 0.02)["verdict"] == "equivalent"
    big = rng.normal(0.08, 0.005, 50)
    assert corrected_compare(big, 800, 200, 0.02)["verdict"] == "different"
    noisy = rng.normal(0.0, 0.08, 50)
    assert corrected_compare(noisy, 800, 200, 0.02)["verdict"] == "inconclusive"
