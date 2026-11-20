# Proxy labels in learning-management-system early warning

Starter repository for the MSc study: how valid are inactivity-based dropout labels when no official
withdrawal record exists, and how much does the choice of label change which students an
early-warning model flags? OULAD supplies the official record. KNUST Moodle data supplies the real case.

**First time here? Do these three things, in this order.**
1. Read the Workflow Guide (file 01 of your pack) from start to finish.
2. Install: `pip install -r requirements.txt`, then `python -m pytest -q` (13 tests must pass).
3. Run the practice study: `python run_study.py --mode demo`. It takes about 5 minutes and needs no real data.

A working demo is your first milestone. Real data comes later, in the order set out in the Workflow Guide.

## Quick start

```bash
python 04_src/00_check_env.py         # what your computer has
python -m pytest -q                   # every rule the study depends on
python run_study.py --mode demo       # the whole study on synthetic data
```

Demo outputs go to `06_results/demo/`. Every demo table and figure says
**SYNTHETIC DEMO DATA - not a result**. Never copy a demo number anywhere.

## Step table

| Step | Script | What it does | Output | Feeds | RQ |
|---|---|---|---|---|---|
| 00 | `00_check_env.py` | Checks Python, packages, memory, data folders | screen | none | none |
| 01 | `01_make_demo_data.py` | Synthetic OULAD-like and KNUST-like files | `02_data/demo/` | practice only | none |
| 02 | `02_check_oulad_files.py` | Real mode: checks the 7 OULAD files, fingerprints them | `02_data/datasheet.md` | Data section | none |
| 03 | `03_adapt_oulad.py` | OULAD into three standard tables | `02_data/processed/<mode>/oulad/` | all | none |
| 04 | `04_adapt_knust.py` | KNUST exports into three standard tables | `02_data/processed/<mode>/knust/` | all | none |
| 05 | `05_lock_plan.py` | Locks labels, margins and seeds before real runs | `03_configs/plan_lock.json` | Methods: lock and deviations | all |
| 06 | `06_build_labels.py` | L0 to L5 and the onset day of each | `labels.csv`, `label_prevalence` | Table: label definitions | RQ1a, RQ1b |
| 07 | `07_build_features.py` | Shared features at days 14, 28, 42 | `features_T*.csv`, `feature_parity` | Methods | all |
| 08 | `08_e2_proxy_validity.py` | E2: agreement, timing, train on proxy and judge on truth (OULAD) | `e2_*` tables | Results 4.3, Figs E2 | RQ1a |
| 09 | `09_e3_label_multiplicity.py` | E3: performance, list overlap, flip rate, channel audit (KNUST) | `e3_*` tables | Results 4.4, Fig E3 | RQ1b |
| 10 | `10_e4_circularity.py` | E4: pre-onset share, active students, ablation, gain against L0 | `e4_*` tables | Results 4.5, Fig E4 | RQ2 |
| 11 | `11_e5_transfer_equivalence.py` | E5: arms A to D, equivalence tests, 20 small draws | `e5_*` tables | Results 4.6, Fig E5 | RQ3 |
| 12 | `12_e6_tabpfn.py` | E6: TabPFN comparator, added after the freeze | `e6_*` tables | Results 4.6 | RQ3 |
| 13 | `13_e7_registrar_check.py` | E7 (optional): PPV and NPV of L1 from the Registrar's counts | `e7_registrar_check` | Results 4.4 or Limitations | RQ1b |
| 14 | `14_tables_figures.py` | All figures, greyscale with one olive highlight | `06_results/<mode>/figures/` | Figures | all |

E1, the rebuild of the baseline (Wu et al., 2026), stays in your existing code
(`scripts/09_wu_comparable_sensitivity.py`). Copy its output into
`07_docs/templates/e1_baseline_rebuild_TEMPLATE.csv` and save it as `06_results/real/tables/e1_baseline_rebuild.csv`.

## How this fits with your existing repository

Your frozen pipeline stays the source of truth for the features and the models it already runs.
Copy this whole folder into your repository as `label_study/`, or keep it as its own repository.
If your frozen code writes the feature tables, set `features.source: external` in `03_configs/config.yaml`.
Step 07 then prints a parity table that compares its own recipe with yours.

## Rules

- **KNUST data never leaves your laptop.** Not to GitHub, Colab, Google Drive, e-mail or any AI tool.
  `.gitignore` blocks `02_data/raw/`, `02_data/processed/` and `00_admin/`.
- **Raw files are read-only.** The lock fingerprints the KNUST exports. A changed file stops every real run.
- **No setting lives in a script.** Every setting is in `03_configs/config.yaml`.
- **After the lock, a change is a deviation.** Log it with `python 04_src/05_lock_plan.py --deviation "reason"`.
- **Tables and figures come from code only.** Never type a number into a table by hand.
- **Never invent, copy or alter data**, labels or Registrar counts.
- **When a step fails:** read the message, fix the cause and re-run that step (Workflow Guide, section S1).
- **AI assistants:** see the Workflow Guide, section S2. Log each real use in `07_docs/AI_USE_LOG.md`.

## Repository visibility

Public while you build and analyse (your supervisor reviews the code). Private once the results are
final and writing starts. Public again when the journal editor asks, or at acceptance, with an archived
Zenodo release. Raw data, KNUST files and admin letters never go to GitHub at any stage.

## Data licences

OULAD: Open University Learning Analytics Dataset, released under CC BY 4.0 (CHECK on the download page).
Cite Kuzilek, Hlosta and Zdrahal (2017), Scientific Data. KNUST Moodle exports: used under the Registrar's
approval of 4 August 2026 and HuSSREC clearance. Not shareable. Only aggregate results are published.
