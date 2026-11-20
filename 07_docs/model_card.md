# Model card (fill after the real run)

- Model: HistGradientBoosting with the frozen settings in `03_configs/config.yaml` (models.hgb).
- Comparators: logistic regression. TabPFN (added after the freeze).
- Intended use: ranking students for adviser contact at a fixed capacity (top 20%) at day 28.
- Not for: automatic decisions about any student. Not for marks, sanctions or records.
- Training data: OULAD (official label L0) and KNUST Moodle exports (proxy labels L1 to L5).
- Known limits: KNUST labels are proxies built from the same activity logs as the features.
  One institution and one presentation. No demographic fields, so no fairness audit on KNUST data
  beyond the access-channel audit.
- Performance: CHECK after the real run (from 06_results/real/tables, never typed by hand).
