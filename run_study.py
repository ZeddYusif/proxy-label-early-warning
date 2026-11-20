"""Run the whole study in order. Works on Windows, macOS and Linux.

  python run_study.py --mode demo            practice run on synthetic data (no keys, no cost, no real data)
  python run_study.py --mode real            real run, after the lock (Workflow Guide, Phases 4 and 5)
  python run_study.py --mode real --from 08  start again from step 08

Each step prints what it did. The run stops at the first error and says what to fix.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = ["00_check_env", "01_make_demo_data", "03_adapt_oulad", "04_adapt_knust", "06_build_labels",
        "07_build_features", "08_e2_proxy_validity", "09_e3_label_multiplicity", "10_e4_circularity",
        "11_e5_transfer_equivalence", "12_e6_tabpfn", "13_e7_registrar_check", "14_tables_figures"]
REAL = ["00_check_env", "02_check_oulad_files", "03_adapt_oulad", "04_adapt_knust", "06_build_labels",
        "07_build_features", "08_e2_proxy_validity", "09_e3_label_multiplicity", "10_e4_circularity",
        "11_e5_transfer_equivalence", "12_e6_tabpfn", "13_e7_registrar_check", "14_tables_figures"]
NO_MODE = {"00_check_env", "01_make_demo_data", "02_check_oulad_files"}

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["demo", "real"], default="demo")
p.add_argument("--from", dest="start", default="00", help="two-digit step number to start from")
a = p.parse_args()
steps = [s for s in (DEMO if a.mode == "demo" else REAL) if s[:2] >= a.start]
if a.mode == "real" and not (ROOT / "03_configs" / "plan_lock.json").exists():
    print("STOPPED: lock the plan first:  python 04_src/05_lock_plan.py   (Workflow Guide, section 4.5)")
    sys.exit(1)
t_all = time.time()
for s in steps:
    cmd = [sys.executable, str(ROOT / "04_src" / f"{s}.py")] + ([] if s in NO_MODE else ["--mode", a.mode])
    print(f"\n=== Step {s} ===", flush=True)
    t0 = time.time()
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        print(f"\nStep {s} failed. Read the message above, fix it, then run:  python run_study.py --mode {a.mode} --from {s[:2]}")
        sys.exit(1)
    print(f"    ({time.time() - t0:.0f} s)")
print(f"\nAll steps finished in {(time.time() - t_all) / 60:.1f} minutes. Results are in 06_results/{a.mode}/")
