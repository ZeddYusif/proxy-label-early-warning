"""Shared helpers: settings, folders, demo/real mode, the lock guard, models and metrics.

Every numbered script imports this file. You rarely need to edit it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "03_configs" / "config.yaml"
WATERMARK = "SYNTHETIC DEMO DATA - not a result"


# ---------------------------------------------------------------- settings
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def parse_mode(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["demo", "real"], default="demo",
                   help="demo = synthetic practice data (default). real = OULAD and KNUST files.")
    if extra:
        extra(p)
    return p.parse_args()


def processed_dir(mode: str, dataset: str) -> Path:
    d = ROOT / load_config()["paths"]["processed"] / mode / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(mode: str, kind: str = "tables") -> Path:
    d = ROOT / load_config()["paths"]["results"] / mode / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def say(msg: str) -> None:
    print(msg, flush=True)


def stop(msg: str) -> None:
    print("\nSTOPPED: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------- lock guard
def require_lock(mode: str) -> None:
    """Real runs are refused until the plan is locked and unchanged."""
    if mode != "real":
        return
    cfg = load_config()
    lock = ROOT / cfg["paths"]["lock_file"]
    if not lock.exists():
        stop("the plan is not locked. Run: python 04_src/05_lock_plan.py   (Workflow Guide, section 4.5)")
    data = json.loads(lock.read_text(encoding="utf-8"))
    if data["config_sha256"] != config_hash():
        stop("config.yaml changed after the lock. If the change is needed, log it with:\n"
             "  python 04_src/05_lock_plan.py --deviation \"what changed and why\"")
    for name, h in data.get("knust_raw_sha256", {}).items():
        f = ROOT / cfg["paths"]["knust_raw"] / name.split("/", 1)[1]
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() != h:
            stop(f"{f.name} changed after the lock. Raw exports are read-only. Restore the original file.")


# ---------------------------------------------------------------- tables
def save_table(df: pd.DataFrame, name: str, mode: str) -> Path:
    out = results_dir(mode, "tables") / f"{name}.csv"
    df = df.copy()
    if mode == "demo":
        df.insert(0, "watermark", WATERMARK)
    df.to_csv(out, index=False)
    md = out.with_suffix(".md")
    head = f"**{WATERMARK}**\n\n" if mode == "demo" else ""
    md.write_text(head + to_markdown(df) + "\n", encoding="utf-8")
    say(f"  wrote {out.relative_to(ROOT)}")
    return out


def to_markdown(df: pd.DataFrame) -> str:
    """A small Markdown writer, so no extra package is needed."""
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def read_table(name: str, mode: str) -> pd.DataFrame:
    p = results_dir(mode, "tables") / f"{name}.csv"
    if not p.exists():
        stop(f"{p.relative_to(ROOT)} not found. Run the script that makes it first (see README step table).")
    df = pd.read_csv(p)
    return df.drop(columns=["watermark"], errors="ignore")


# ---------------------------------------------------------------- models
class AllMissingGuard:
    """Fills a column that is entirely missing in the training data (for example scores before the
    first quiz) with 0, so the tree model does not fail. Other missing values stay missing."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cols_ = [c for c in X.columns if X[c].isna().all()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cols_:
            X[c] = 0.0
        return X

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def make_model(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = load_config()["models"]
    if kind == "hgb":
        return make_pipeline(AllMissingGuard(), HistGradientBoostingClassifier(random_state=seed, **cfg["hgb"]))
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(random_state=seed, **cfg["logreg"]))
    raise ValueError(kind)


# ---------------------------------------------------------------- metrics
def top_mask(scores: np.ndarray, frac: float, seed: int = 42) -> np.ndarray:
    """Flag the top `frac` of students by risk. Ties are broken at random with a fixed seed."""
    n = len(scores)
    k = int(round(frac * n))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -np.asarray(scores)))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def precision_recall_at(y: np.ndarray, scores: np.ndarray, frac: float, seed: int = 42):
    m = top_mask(scores, frac, seed)
    tp = np.logical_and(m, y == 1).sum()
    prec = tp / m.sum() if m.sum() else float("nan")
    rec = tp / (y == 1).sum() if (y == 1).sum() else float("nan")
    return float(prec), float(rec)


def ranking_metrics(y, scores, frac):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return dict(pr_auc=np.nan, roc_auc=np.nan, precision_at=np.nan, recall_at=np.nan)
    p, r = precision_recall_at(y, scores, frac)
    return dict(pr_auc=average_precision_score(y, scores), roc_auc=roc_auc_score(y, scores),
                precision_at=p, recall_at=r)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, centre - half, centre + half)


def load_features(mode: str, dataset: str, window: int) -> pd.DataFrame:
    cfg = load_config()
    if cfg["features"]["source"] == "external" and mode == "real":
        p = ROOT / cfg["features"]["external_pattern"].format(dataset=dataset, window=window)
    else:
        p = processed_dir(mode, dataset) / f"features_T{window}.csv"
    if not p.exists():
        stop(f"{p} not found. Run 04_src/07_build_features.py --mode {mode} first.")
    return pd.read_csv(p)


def load_labels(mode: str, dataset: str) -> pd.DataFrame:
    p = processed_dir(mode, dataset) / "labels.csv"
    if not p.exists():
        stop(f"{p} not found. Run 04_src/06_build_labels.py --mode {mode} first.")
    return pd.read_csv(p)


def cohort(mode: str, dataset: str, window: int) -> pd.DataFrame:
    """Features joined to labels for students still registered at the prediction day."""
    feats = load_features(mode, dataset, window)
    labs = load_labels(mode, dataset)
    df = feats.merge(labs, on=["student_key", "course", "presentation"], how="inner")
    if "unreg_day" in df:
        df = df[~(df["unreg_day"] < window)].copy()
    return df.reset_index(drop=True)


def label_columns(dataset: str) -> list[str]:
    cfg = load_config()["labels"]
    if dataset == "oulad":
        return [f"{p}_{v}" for v in cfg["oulad_versions"] for p in cfg["proxies"]]
    return list(cfg["proxies"])
