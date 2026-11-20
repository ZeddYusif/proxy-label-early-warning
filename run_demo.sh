#!/usr/bin/env bash
# The whole study on synthetic data. No keys, no cost, no real data.
# Windows users: run  python run_study.py --mode demo  instead.
set -e
cd "$(dirname "$0")"
python -m pytest -q
python run_study.py --mode demo
