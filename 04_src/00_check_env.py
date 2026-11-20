"""Step 00. Check your computer before anything else.

Command:  python 04_src/00_check_env.py
Prints:   Python version, package versions, memory, free disk, and which data folders hold files.
Next:     bash run_demo.sh   (or the Colab notebook, Part A)
"""
import importlib
import platform
import shutil
import sys

from common import ROOT, load_config, say

say(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
if sys.version_info < (3, 10):
    say("  WARNING: Python 3.10 or newer is needed. Install a newer Python (Workflow Guide, section 2.4).")

ok = True
for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml"]:
    try:
        m = importlib.import_module(pkg)
        say(f"  {pkg:<11} {getattr(m, '__version__', '?')}")
    except ImportError:
        ok = False
        say(f"  {pkg:<11} MISSING. Run: pip install -r requirements.txt")
try:
    import tabpfn  # noqa: F401
    say("  tabpfn      installed (optional, used only in E6)")
except ImportError:
    say("  tabpfn      not installed (optional, needed only for E6 real runs)")

try:
    import os
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    say(f"Memory: {mem_gb:.1f} GB" + ("  (8 GB or more is comfortable for the full OULAD files)" if mem_gb < 8 else ""))
except (ValueError, OSError, AttributeError):
    say("Memory: could not read on this system (fine on Windows)")
say(f"Free disk: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB  (the OULAD files need about 0.5 GB)")

cfg = load_config()
for name in ["oulad_raw", "knust_raw"]:
    folder = ROOT / cfg["paths"][name]
    files = sorted(p.name for p in folder.glob("*.csv")) if folder.exists() else []
    say(f"{name}: {len(files)} CSV file(s) in {cfg['paths'][name]}")

say("\nAll core packages present. Next: bash run_demo.sh" if ok else "\nFix the missing packages, then run this check again.")
