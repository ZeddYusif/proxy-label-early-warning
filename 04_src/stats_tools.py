"""Equivalence and agreement statistics.

corrected_compare: compares two models over repeated cross-validation folds with the
Nadeau and Bengio (2003) corrected variance, then applies the three-word rule:
  equivalent    the 90% interval lies inside the margin (two one-sided tests at 5%)
  different     not equivalent, and the 95% interval excludes zero
  inconclusive  neither
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_compare(diffs, n_train: float, n_test: float, margin: float) -> dict:
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 2:
        return dict(n_folds=J, mean_diff=np.nan, ci90_low=np.nan, ci90_high=np.nan,
                    ci95_low=np.nan, ci95_high=np.nan, verdict="inconclusive")
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    df = J - 1
    m = d.mean()
    t90 = stats.t.ppf(0.95, df)
    t95 = stats.t.ppf(0.975, df)
    out = dict(n_folds=J, mean_diff=m, ci90_low=m - t90 * se, ci90_high=m + t90 * se,
               ci95_low=m - t95 * se, ci95_high=m + t95 * se)
    out["verdict"] = verdict(out, margin)
    return out


def bootstrap_compare(values_a, values_b, margin: float) -> dict:
    """For paired bootstrap differences (a minus b) over students."""
    d = np.asarray(values_a) - np.asarray(values_b)
    d = d[~np.isnan(d)]
    out = dict(n_folds=len(d), mean_diff=float(np.mean(d)),
               ci90_low=float(np.percentile(d, 5)), ci90_high=float(np.percentile(d, 95)),
               ci95_low=float(np.percentile(d, 2.5)), ci95_high=float(np.percentile(d, 97.5)))
    out["verdict"] = verdict(out, margin)
    return out


def verdict(r: dict, margin: float) -> str:
    if r["ci90_low"] > -margin and r["ci90_high"] < margin:
        return "equivalent"
    if r["ci95_low"] > 0 or r["ci95_high"] < 0:
        return "different"
    return "inconclusive"


def kappa_from_counts(tp, fp, fn, tn) -> float:
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def kappa_band(k: float, acceptable: float, weak: float) -> str:
    if np.isnan(k):
        return "not computable"
    if k >= acceptable:
        return "acceptable proxy"
    if k >= weak:
        return "weak"
    return "not valid"
