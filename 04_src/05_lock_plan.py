"""Step 05. Lock the plan before any real run under the new labels.

Command:  python 04_src/05_lock_plan.py                       (first lock)
          python 04_src/05_lock_plan.py --show                (print the lock)
          python 04_src/05_lock_plan.py --deviation "reason"  (record a needed change after the lock)
Writes:   03_configs/plan_lock.json  and, for deviations, a dated line in 07_docs/deviation_log.md
Then:     git add 03_configs/plan_lock.json && git commit -m "Lock plan v2" && git tag v2-lock
          Record the commit hash (git rev-parse HEAD) and the date. Both go into your Methods section.

Why: the lock proves that the labels, margins and seeds were fixed before you saw any result.
Every real-mode script refuses to run if config.yaml or a KNUST raw file changed after the lock.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess

from common import ROOT, config_hash, load_config, say, stop

p = argparse.ArgumentParser()
p.add_argument("--show", action="store_true")
p.add_argument("--deviation", type=str, default=None)
args = p.parse_args()
cfg = load_config()
lock_path = ROOT / cfg["paths"]["lock_file"]


def file_hashes():
    out = {}
    folder = ROOT / cfg["paths"]["knust_raw"]
    for f in sorted(folder.glob("*.csv")):
        out[f"knust/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOT A GIT REPOSITORY YET"


if args.show:
    if not lock_path.exists():
        stop("no lock yet.")
    say(lock_path.read_text(encoding="utf-8"))
    raise SystemExit

now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
if lock_path.exists():
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not args.deviation:
        if lock["config_sha256"] == config_hash():
            say("The plan is already locked and config.yaml is unchanged. Nothing to do.")
            raise SystemExit
        stop("config.yaml changed after the lock. If the change is needed, run again with --deviation \"reason\".")
    old = lock["config_sha256"]
    lock["config_sha256"] = config_hash()          # raw-file fingerprints are never renewed here
    lock.setdefault("deviations", []).append({"date": now, "old_config": old, "new_config": lock["config_sha256"],
                                              "reason": args.deviation})
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    log = ROOT / cfg["paths"]["deviation_log"]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"| {now} | {args.deviation} | {old[:12]} | {lock['config_sha256'][:12]} |\n")
    say(f"Deviation recorded in {log.relative_to(ROOT)}. Report it in the manuscript (Methods, lock and deviations).")
    raise SystemExit

if not file_hashes():
    stop("no KNUST exports in 02_data/raw/knust/. Put the three files there first, so the lock fingerprints them.")
lock = {
    "locked_at": now,
    "config_sha256": config_hash(),
    "git_commit_at_lock": git_commit(),
    "python": platform.python_version(),
    "labels": cfg["labels"],
    "windows": cfg["windows"],
    "margins": cfg["margins"],
    "kappa_bands": cfg["kappa_bands"],
    "seeds": cfg["seeds"],
    "cv": cfg["cv"],
    "capacity": cfg["capacity"],
    "models": cfg["models"],
    "tabpfn": cfg["tabpfn"],
    "knust_raw_sha256": file_hashes(),
    "interpretation_table": "07_docs/interpretation_table.md",
    "pre_lock_results": "All results in the draft of 18 November 2026 are pre-lock.",
}
lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
say(f"Locked at {now}. config sha256 {lock['config_sha256'][:12]}...  KNUST files fingerprinted: {len(lock['knust_raw_sha256'])}")
say("Now commit and tag:  git add 03_configs/plan_lock.json && git commit -m \"Lock plan v2\" && git tag v2-lock")
say("Then record the output of  git rev-parse HEAD  and today's date for your Methods section.")
