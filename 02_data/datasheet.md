# Datasheet

## OULAD
- Source: Open University Learning Analytics Dataset (Kuzilek, Hlosta and Zdrahal, 2017, Scientific Data).
- Licence: CC BY 4.0 (CHECK on the download page). Downloaded on: CHECK.
- Files and fingerprints: written below by `04_src/02_check_oulad_files.py`.

## KNUST Moodle exports (PRIVATE)
- Supplied by: the university, pseudonymised (P00001 style codes). The key stays with the university.
- Approvals: Registrar release of 4 August 2026. HuSSREC clearance (CHECK: reference).
- Files: activity log, gradebook, quiz attempts. Fingerprints are kept in `03_configs/plan_lock.json`.
- Known limits: no enrolment table, no year of study, no forum or assignment records, no demographics,
  no due dates, no withdrawal record. The gradebook covers 18 January to 12 May 2026 only.

## OULAD files checked 2026-11-22

| File | Rows | SHA-256 |
|---|---|---|
| courses.csv | 22 | 4f16eee7454b15e109b0a21a0e43be820e6846ed6f9301bb7feb5ab5ad737a75 |
| assessments.csv | 206 | 8cc738fb88ad760571d6f2a23059bfee0ffcae3bcd830514c9cbd5c6d5a046f1 |
| vle.csv | 6,364 | d1b28303dea802ad87b4484e1196e878e06824850b9a4fe8aa34693439fe87e9 |
| studentInfo.csv | 32,593 | 7e6f3e474a5eee00639d2a414a6c7e928745823c2d2c2563ca1780145f99b0d6 |
| studentRegistration.csv | 32,593 | 0d32676285372aaf2e7a80304e5b274b4fba24313e2ca4c04317225e1ec90170 |
| studentAssessment.csv | 173,912 | fd5320786328d05af841ee7dd4b5871b9dada3b9fe9d6a3642b2f42635510a6e |
| studentVle.csv | 10,655,280 | 52668253d876c5becbcb72185977152700cecab2942aca807fecc3dd54b937f0 |
