# Everyday Updates

Purpose
- This file tracks day-by-day project progress in one place.
- It records what was implemented, what was pushed to GitHub, and what is pending.
- It should be updated at the end of each work session.

How to update this file each day
- Add a new date section at the top.
- Record technical work done, repository actions, and blockers.
- Keep entries short, factual, and traceable.

## 2026-03-18

Detailed two-CSV discussion record
- Primary files reviewed:
  - 1.unsorted_combined_measurements_data.csv
  - 2.aggregated_measurements_data.csv
- Analysis mode used:
  - Full-file profiling (not sample-only), chunked reads for memory safety.
  - Header/schema inspection, row counts, file sizes, type/range checks, missing-value checks, duplicate-key checks, time parsing checks, and outlier extraction.

Two-file baseline facts
- Unsorted raw file:
  - Size: 1,952,502,329 bytes
  - Columns: 81
  - Rows: 2,313,903
  - Character: TTN-style raw payload and metadata export.
- Aggregated file:
  - Size: 297,698,840 bytes
  - Columns: 20
  - Rows: 1,715,869
  - Character: compact modeling table for ML.

Schema and identity conclusions
- Raw unsorted file includes wide metadata (`rx_metadata_*`, network IDs, device version fields).
- Aggregated file keeps modeling columns (`co2`, `humidity`, `pm25`, `pressure`, `temperature`, `rssi`, `snr`, `distance`, `exp_pl`, etc.).
- In unsorted raw data, `device_id` is not the true per-node field.
- True per-node identity in unsorted raw is `end_device_ids_device_id`.
- Aggregated dataset has normalized IDs (`ED0` to `ED5`).

Time and ordering findings
- Shared broad time coverage across both files:
  - Start: 2024-09-26 11:00:52.541686+00:00
  - End: 2025-05-22 14:56:11.322763+00:00
- Time parse failures:
  - Unsorted raw: 2
  - Aggregated: 0
- Aggregated file is not globally monotonic by time, so explicit sort is required before split/training.

Missingness and duplicate checks
- Unsorted raw:
  - Missing snr: 2,044
  - Missing uplink_message_f_cnt: 20
  - Duplicate key rows on [time, end_device_ids_device_id, uplink_message_f_cnt]: 1,032
- Aggregated:
  - Missing snr: 1,427
  - Missing f_count: 19
  - Duplicate key rows on [time, device_id, f_count]: 0

Critical data quality findings discussed
- Pressure scaling issue in both files:
  - Stored pressure values are in compressed scale (~299 to ~342 typical).
  - Corrected pressure rule: pressure_hPa = pressure * 3.125.
  - Validation check: ~99.998% of aggregated rows fall into 800 to 1200 hPa after scaling.
- Deterministic bad rows in aggregated file:
  - Total: 33 rows
  - Pattern A: 19 rows (`co2=21547`, `humidity=156.65`, `temperature=174.90`, `pressure=3.21`, `pm25=33.93`)
  - Pattern B: 2 rows (`co2=16724`, `humidity=210.53`, `temperature=110.76`, `pressure=317.45`, `pm25=125.57`)
  - Pattern C: 12 rows (`co2=0`, `humidity=0`, `temperature=0`, `pressure=508.90`, `pm25=0`)
  - Decision taken: remove all 33 rows (no imputation).

Final decisions recorded from discussion
- Use aggregated file as default model-training source.
- Keep unsorted raw file as source-of-truth for trace-back and audit.
- Mandatory preprocessing agreed:
  - Remove 33 known anomaly rows.
  - Apply pressure correction (x3.125).
  - Drop rows with null `snr` and `f_count` for supervised link-quality training.
  - Sort by `time` before train/validation/test splitting.
  - Log row counts before and after each cleaning step.

Code integration completed in train_model.py
- Real data path enabled by default (`2.aggregated_measurements_data.csv`).
- Deterministic anomaly filters (Pattern A/B/C) implemented.
- Pressure correction implemented.
- Null filtering (`snr`, `f_count`) implemented.
- Time parse and sort implemented when `time` exists.
- Label generation from `rssi` and `snr` implemented when `link_state` is absent.
- Loader now prints preprocessing summary (raw rows, removed anomalies, dropped null rows, final rows).

Documentation updates completed
- Added full dataset audit section to `FederatedTinyML/README.md`.
- Added corresponding summary to root `README.md`.
- Added daily log framework in `Everyday Updates.md` for ongoing tracking.

Repository and sync work completed
- First push attempt failed because CSV files exceeded GitHub 100 MB limit.
- Switched dataset upload method to Git LFS.
- Successfully pushed datasets and README audit update to main:
  - Commit: 111e204
  - Message: Add datasets via Git LFS and dataset audit README
- Cleaned and resynced temporary upload clone:
  - Branch main aligned with origin/main
  - Local and remote both at 111e204

Current blockers
- Runtime environment is missing TensorFlow in active Python environment.
- Result: loader smoke-test command failed at import stage, not due to data logic.

Next actions
- Install Python dependencies in active environment (tensorflow, pandas, numpy, scikit-learn).
- Run loader smoke test again.
- Run full training pipeline on cleaned aggregated dataset.
- Save model artifacts and verify class distribution and evaluation metrics.

## 2026-03-17

Repository work completed
- Revised literature review and image links.
- Added and adjusted project files through multiple uploads.
- Restored repository root README.md after structure changes.
- Standardized repository structure around federated_tinyml and pics.

Notable commits
- 51ccff3: Revise literature review and update image links
- f301d86: Update LITERATURE_REVIEW.md by removing an image link
- a999ade: Add image links to LITERATURE_REVIEW.md
- 46540f7: Update federated_tinyml and pics
- 9a34463: Restore README.md at repository root
- 6d0458f: Keep only federated_tinyml and pics folders at repo root

## 2026-03-14

Repository hygiene and structure updates
- Continued cleanup and structure normalization in repo.
- Removed duplicate folder patterns and kept intended root layout.
- Restored required files after cleanup where needed.

Notable commits
- d9c6481: Delete unwanted PDF in pics
- a87f922: Restore kept-folder files after root cleanup
- 42a75c9: Keep only federated_tinyml, pics, and README at repo root
- ee028c1: Remove duplicate FederatedTinyML folder
- c1ba17c: Update federated_tinyml

## 2026-03-13

Project bootstrap and initial uploads
- Added thesis project files and datasets.
- Added and updated project README files.

Notable commits
- 039c7ee: Add local thesis project files and datasets
- c6ddf2d: Update README.md
- 57f162a: Add README for Federated TinyML project

## 2026-03-11

Documentation and reference updates
- Improved README image linking and literature references.
- Initialized and adjusted pics folder state.

Notable commits
- 75b130d: Fix image links and add literature references
- 39e4c50: Fix image link formatting in README.md
- 6034b24: Enhance README with image links
- c7d1bc3: Create pics
- 65f01b1: Delete pics

## 2026-03-09

Project initialization
- Initial repository commit completed.

Notable commit
- aa0a0d3: Initial commit
