# Falsification log

One line every time a result surprised you, a check failed, or you tried something and dropped it.
Dropped attempts are evidence too. Never delete a line.

| Date | Step | What you expected | What happened | What you did | Commit |
|---|---|---|---|---|---|
| 2026-11-22 | 04 adapt KNUST, real | The adapter would read the real activity log | Crashed: cannot subtract tz-naive and tz-aware times | Parsed times as UTC and dropped the time zone, for the activity log and the quiz attempts | f3d06b4 |
| 2026-11-22 | 04 adapt KNUST, real | The adapter would read the real gradebook | The gradebook was long (one row per student per quiz), not wide as the adapter expected | Pivoted it to one row per student | f3d06b4 |
| 2026-11-22 | 01 make demo data | The demo generator would produce data matching the real gradebook | It hard-coded quiz names that no longer matched the gradebook | Made it read the quiz names from the config | 1abb12a |
| 2026-11-22 | 04 adapt KNUST, real | The cohort would match the expected 4,450 students | The count was 4,391, so 59 students were missing. They had no events in the window and no grades | Added the student IDs from the unfiltered log | f3d06b4 |
| 2026-11-22 | config, frozen HGB values | The HGB model would run with the frozen settings | The config still held the starter's values (depth 3, no l2, early stopping on) | Corrected it to the frozen values | f3d06b4 |
| 2026-11-22 | export frozen features | The export script would find the config and be tracked in the repository | The default config path was wrong, and the script was first placed in a git-ignored folder | Fixed the default path and moved the script to 04_src | 255375b |
| 2026-11-20 | old repository (Shapalert_GH) | Only code would be public and no tag would exist | It was pushed publicly with result tables and a mistaken v2-lock tag (Phase 4 commands run in the wrong folder). A history audit on 22 Nov found aggregate tables only, no raw or per-student files | Deleted the v2-lock tag on 20 Nov; set it to private on 2026-11-20 ; first public push on 2026-11-20  | 964c8af (old repo) |
| 2026-11-20 | GitHub README, study repository | The starter README would stay intact | Two web edits (a917e12, 81eef17) emptied it | Restored it with a new commit on 2026-11-22 | df261a4 |