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

Project work completed
- Performed full-file profiling of both datasets:
  - 1.unsorted_combined_measurements_data.csv
  - 2.aggregated_measurements_data.csv
- Verified schema, row counts, missingness, duplicates, value ranges, and time coverage.
- Confirmed pressure scaling issue and validated correction rule:
  - pressure_hPa = pressure * 3.125
- Identified deterministic anomaly families in aggregated data (total 33 rows):
  - Pattern A: 19 rows
  - Pattern B: 2 rows
  - Pattern C: 12 rows
- Confirmed device identity rule in raw data:
  - Use end_device_ids_device_id for per-node identity, not device_id.
- Added dataset audit summary into FederatedTinyML/README.md.
- Updated train_model.py to use real aggregated dataset by default with safe preprocessing:
  - Uses 2.aggregated_measurements_data.csv by default
  - Drops 33 known bad rows
  - Applies pressure correction (x3.125)
  - Drops rows with null snr and f_count
  - Sorts by time when available
  - Derives labels from RSSI/SNR if link_state is absent

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
