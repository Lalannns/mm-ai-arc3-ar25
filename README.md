# AR25 Experiment QA Dataset

## Dataset artifacts

- Parquet dataset: `data/parquet/ar25_experiment_qa_corrected.parquet`
- Source-to-canonical-record mapping: `data/parquet/ar25_experiment_qa_provenance.json`
- Validation reports: `reports/experiment_qa_validation_report.json` and `reports/experiment_qa_validation_report.md`

The builder reads the immutable raw experiment logs and frame files under `data/raw/exploration/`. It does not modify or delete raw evidence.

## Integrity and provenance rules

- Frame hashes are SHA-256 over a versioned canonical payload containing the 2D array dtype, shape, and C-order pixel bytes. NPY headers and memory layout do not affect the hash.
- Trajectory identity includes the initial frame, ordered target actions and action data, every ordered after-frame observation, final frame, outcome, final state, prefix actions, and level values with provenance.
- Every raw experiment is retained in the provenance sidecar, including records collapsed into a canonical QA row. Each source entry identifies the JSONL file, raw line index, run, experiment, seed, timestamp, prefix, and recorded outcome.
- Target actions and prefix actions are separate. Prefix actions are never counted as target steps.
- Historical starting levels and prefix actions that were not recorded remain `null` / `unknown`. The builder does not infer them from the first post-action level.
- Ending levels are taken from an explicit top-level field when present, otherwise from the final step's post-action observation.
- `total_changed_cells` means the sum of per-step full-frame changed-cell counts. A cell may be counted more than once if it changes repeatedly.
- Outcomes are recomputed from saved pixel observations where possible. Historical `level_progress` labels without a recorded starting level are retained as `recorded_unverified`; they are not presented as independently verified.
- The validator checks Draft 2020-12 schema conformance, source mapping, canonical hashes, trajectory signatures, frame chaining, pixel-change metrics, outcome consistency, and answer claims that are explicitly machine-checkable. It does not claim unrestricted natural-language semantic verification.

## Current archived raw snapshot: preflight findings

The archived snapshot contains 16 exploration directories and 1,300 raw experiment records. An in-memory rebuild with the corrected trajectory identity produced 1,021 canonical QA records, with all 1,300 source records mapped in the provenance sidecar (279 source records collapsed into existing canonical records). This is a preflight calculation; regenerate the Parquet file and run the validator locally before treating the rebuilt artifact as accepted.

| Derived outcome | Canonical records |
|---|---:|
| `changed` | 873 |
| `no_visible_change` | 142 |
| `level_progress` (recorded, unverified) | 6 |
| **Total** | **1,021** |

The old raw logs omit top-level starting levels and prefix-action provenance. Accordingly, all 1,021 historical canonical records have unknown starting-level and prefix provenance. Ending levels are available from the final target step. The 6 unique `level_progress` rows retain that recorded label as unverified because the historical starting level is unavailable.

A historical outcome-label defect was also identified: 42 raw records in two repeated batch runs were labeled `changed_then_returned_to_start`, but their saved initial and final pixel arrays differ. The corrected builder derives their visual outcome from the saved observations instead of trusting that inconsistent label. The updated exploration scripts copy observation arrays before later environment steps to prevent mutable-buffer aliasing from corrupting future classifications.

## Research limitations

- QA rows are experiment/trajectory observations, not independent ARC reasoning problems.
- Repeated initial visual states and convergent trajectories create dependence. Random row splits can leak related evidence; group splits by initial visual state and experiment family/trajectory.
- `estimate_counter_cells` is a heuristic based on the top HUD strip, not a verified internal game counter.
- This dataset documents observed experiment sequences. It is not proof that an autonomous VLM can solve ARC3_AR25.

## Rebuild and validate

Run from the repository root after installing `requirements-dataset.txt` in the project virtual environment:

```zsh
.venv/bin/python scripts/16_build_experiment_qa_parquet.py
.venv/bin/python scripts/validate_experiment_qa.py
.venv/bin/python -m pytest -q
```

The validator exits non-zero on critical errors. Do not commit or push until the Parquet rebuild, provenance checks, validator, and tests have passed locally.
