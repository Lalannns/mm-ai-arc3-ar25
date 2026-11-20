# AR25 dataset validation report

Generated: 2026-11-20 02:31 UTC  
Command: `python scripts/validate_dataset.py`  
Status: **PASS** (0 errors, 3 warnings)

## Raw data (`data/raw/`)

| Item | Count |
|---|---|
| sequential trajectories | 1 |
| sequential steps | 5 |
| single action transitions | 11 |
| episode log steps | 37 |
| npy frame files | 33 |
| orphan npy files | 0 |

## Processed dataset `data/processed/ar25_transitions.parquet`

| Item | Value |
|---|---|
| rows | 5 |
| unique transitions | 5 |
| duplicate transitions | 0 |
| rows with no visible change | 0 |
| actions | ACTION1: 1, ACTION2: 1, ACTION3: 1, ACTION4: 1, ACTION5: 1 |
| sources | sequences.jsonl: 5 |
| reasoning | lisa:lisa-pro: 5 |

## Processed dataset `data/processed/ar25_transitions_full.parquet`

| Item | Value |
|---|---|
| rows | 262 |
| unique transitions | 146 |
| duplicate transitions | 116 |
| rows with no visible change | 65 |
| actions | ACTION1: 57, ACTION2: 68, ACTION3: 37, ACTION4: 24, ACTION5: 19, ACTION6: 40, ACTION7: 17 |
| sources | exploration/20261120T014608Z: 140, exploration/level2_20261120T020820Z: 70, logs/ep_001.json: 6, logs/ep_ai_001.json: 30, sequences.jsonl: 5, transitions.jsonl: 11 |
| reasoning | observed_diff: 262 |

## Checks

| Check | Errors | Warnings |
|---|---|---|
| missing_file | 0 | 0 |
| unreadable_file | 0 | 0 |
| malformed_json | 0 | 0 |
| missing_field | 0 | 0 |
| dimensions | 0 | 0 |
| value_range | 0 | 0 |
| step_order | 0 | 0 |
| broken_sequence | 0 | 0 |
| incomplete_transition | 0 | 1 |
| duplicate_record | 0 | 0 |
| duplicate_transition | 0 | 1 |
| duplicate_flag | 0 | 0 |
| changed_pixels_mismatch | 0 | 0 |
| hash_mismatch | 0 | 0 |
| schema | 0 | 0 |
| orphan_file | 0 | 0 |
| legacy_format | 0 | 1 |

## Findings

| Level | Check | Target | Message |
|---|---|---|---|
| WARNING | legacy_format | `data/raw/trajectories/transitions.jsonl:20261119T100908020911Z` | uses before_file/after_file instead of *_files lists |
| WARNING | incomplete_transition | `data/raw/logs/ep_001.json:step6` | last step has no next_grid_state, so its after frame is unknown |
| WARNING | duplicate_transition | `data/processed/ar25_transitions_full.parquet` | 116 rows repeat an identical (before, action, after) transition |
