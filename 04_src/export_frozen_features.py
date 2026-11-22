
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

FEATURES = [
    "clicks_total_z", "active_days_z", "active_ratio", "clicks_per_active_day_z",
    "gap_frac", "first_click_frac", "clicks_last_week_z", "click_trend_z",
    "n_early_assess_z", "has_early_assess", "early_score_mean",
]
NO_BLANK = [f for f in FEATURES if f != "early_score_mean"]
WINDOWS = [14, 28, 42]

# dataset -> (old input prefix, output prefix). Edit the output prefix if
# config.yaml's pattern differs.
DATASETS = {"oulad": ("oulad", "oulad"), "knust": ("ghana", "knust")}

EXPECTED = {
    "oulad": {14: 28128, 28: 27553, 42: 27034},
    "knust": {14: 4450, 28: 4450, 42: 4450},
}


def need(df, cols, path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"{path.name}: missing column(s) {missing}. Columns present: {list(df.columns)}")


def build_oulad(df, path):
    need(df, ["id_student", "course"], path)
    sid = df["id_student"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    course = df["course"].astype(str).str.strip()
    if "code_module" in df.columns and "code_presentation" in df.columns:
        module = df["code_module"].astype(str).str.strip()
        pres = df["code_presentation"].astype(str).str.strip()
    else:  # course looks like AAA_2013J
        split = course.str.split("_", n=1, expand=True)
        module, pres = split[0], split[1]
    return pd.DataFrame({"student_key": module + "_" + pres + "_" + sid,
                         "course": course, "presentation": pres})


def build_knust(df, path):
    need(df, ["pseudo_id"], path)
    n = len(df)
    return pd.DataFrame({"student_key": df["pseudo_id"].astype(str).str.strip(),
                         "course": ["KNUST_2026"] * n, "presentation": ["2026"] * n})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_config_features(config_path, problems):
    """Compare FEATURES with features.shared in config.yaml (order-insensitive, and
    also reports order differences)."""
    if not config_path.exists():
        problems.append(f"config not found at {config_path} (run from the starter root or pass --config)")
        return
    try:
        import yaml
    except ImportError:
        problems.append("PyYAML missing: pip install pyyaml (needed for the config feature check)")
        return
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        shared = list(cfg["features"]["shared"])
    except (KeyError, TypeError):
        problems.append(f"{config_path}: no features.shared list found")
        return
    if set(shared) != set(FEATURES):
        problems.append("features.shared differs from the script: "
                        f"only in config {sorted(set(shared) - set(FEATURES))}, "
                        f"only in script {sorted(set(FEATURES) - set(shared))}")
    elif shared != FEATURES:
        problems.append("features.shared has the same names but a different order from the script")
    else:
        print(f"Feature list matches {config_path.name} features.shared ({len(shared)} features).")


def check_keys(ds, T, out, students_root, problems):
    """Compare exported keys with the starter's students.csv, if it exists yet."""
    sp = students_root / ds / "students.csv"
    if not sp.exists():
        print(f"  key check skipped for {ds}: {sp.as_posix()} not found yet "
              f"(rerun with --check-only after step 04)")
        return
    st = pd.read_csv(sp, dtype={"student_key": str}, usecols=["student_key"])
    starter, mine = set(st["student_key"]), set(out["student_key"])
    not_in_starter = len(mine - starter)
    only_starter = len(starter - mine)
    if ds == "knust":
        ok = not_in_starter == 0 and only_starter == 0
        print(f"  key check {ds} T{T}: {len(mine):,} exported, {len(starter):,} in starter, "
              f"{not_in_starter} not in starter, {only_starter} only in starter  [{'OK' if ok else 'MISMATCH'}]")
        if not ok:
            problems.append(f"{ds} T{T}: key sets are not identical")
    else:
        ok = not_in_starter == 0
        print(f"  key check {ds} T{T}: {len(mine):,} exported, {len(starter):,} in starter, "
              f"{not_in_starter} not in starter, {only_starter:,} only in starter (expected: "
              f"unregistered before window)  [{'OK' if ok else 'MISMATCH'}]")
        if not ok:
            problems.append(f"{ds} T{T}: {not_in_starter} exported keys are missing from the starter's list")


def verify(ds, T, out, dst, students_root, problems):
    n, exp = len(out), EXPECTED[ds][T]
    n_blank = int(out["early_score_mean"].isna().sum())
    print(f"{dst.as_posix()}: {n:,} rows, {n_blank:,} empty early_score_mean  "
          f"[{'OK' if n == exp else f'MISMATCH (expected {exp:,})'}]")
    if n != exp:
        problems.append(f"{dst.name}: {n} rows, expected {exp}")
    dup = int(out["student_key"].duplicated().sum())
    if dup:
        problems.append(f"{dst.name}: {dup} duplicate student_key")
    blanks = out[NO_BLANK].isna().sum()
    blanks = blanks[blanks > 0]
    if len(blanks):
        problems.append(f"{dst.name}: empty values in {blanks.to_dict()} (only early_score_mean may be empty)")
    check_keys(ds, T, out, students_root, problems)
    print(f"  sha256 {sha256(dst)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=r"C:\Users\LENOVO\Yussif_Sharpalert\data\processed")
    ap.add_argument("--out-dir", default="02_data/processed/external")
    ap.add_argument("--students-root", default="02_data/processed/real",
                    help="folder holding {dataset}/students.csv written by starter step 04")
    ap.add_argument("--config", default="03_configs/config.yaml")
    ap.add_argument("--check-only", action="store_true",
                    help="do not export; re-check the files already in --out-dir")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    students_root = Path(args.students_root)
    problems = []

    check_config_features(Path(args.config), problems)
    if not args.check_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    for ds, (in_prefix, out_prefix) in DATASETS.items():
        for T in WINDOWS:
            dst = out_dir / f"{out_prefix}_T{T}.csv"
            if args.check_only:
                if not dst.exists():
                    sys.exit(f"Exported file not found: {dst}")
                out = pd.read_csv(dst, dtype={"student_key": str, "course": str, "presentation": str})
            else:
                src = in_dir / f"{in_prefix}_w{T}.csv"
                if not src.exists():
                    sys.exit(f"Input not found: {src}")
                df = pd.read_csv(src, dtype={"id_student": str, "pseudo_id": str,
                                             "code_module": str, "code_presentation": str,
                                             "course": str})
                need(df, FEATURES, src)
                keys = build_oulad(df, src) if ds == "oulad" else build_knust(df, src)
                out = pd.concat([keys.reset_index(drop=True),
                                 df[FEATURES].reset_index(drop=True)], axis=1)
                out.to_csv(dst, index=False, na_rep="")  # empty early_score_mean stays empty
            verify(ds, T, out, dst, students_root, problems)

    if problems:
        print("\nCHECK THESE:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nAll checks passed. Record the sha256 values in your Methods notes.")


if __name__ == "__main__":
    main()