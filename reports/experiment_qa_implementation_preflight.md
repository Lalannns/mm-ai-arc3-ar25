# AR25 Experiment QA implementation preflight

**Scope:** archived source snapshot at baseline commit `41879b1a5bb4572dad0eeb4f8c5ef3cd0f7cbb10`.

## Results from the corrected in-memory rebuild

| Check | Result |
|---|---:|
| Raw exploration directories | 16 |
| Raw JSONL experiment records | 1,300 |
| Canonical trajectory QA records | 1,021 |
| Collapsed source records retained in provenance mapping | 279 |
| Source records independently re-read and signature-checked | 1,300 |
| JSON Schema / row evidence errors | 0 |
| Provenance mapping errors | 0 |
| Historical `level_progress` records with unknown start | 6 |
| Historical rows with unknown start and prefix provenance | 1,021 |
| Regression tests | 16 passed |

## Derived canonical outcome counts

| Outcome | Records |
|---|---:|
| `changed` | 873 |
| `no_visible_change` | 142 |
| `level_progress` (recorded, unverified) | 6 |
| **Total** | **1,021** |

The archived raw logs contain 42 `changed_then_returned_to_start` labels whose saved initial and final pixel arrays differ. The corrected builder recomputes the visual outcome from saved frames and preserves the original label in `recorded_outcome` and in the source mapping.

## Limitations and pending local verification

- Historical raw records do not contain reliable top-level starting levels or prefix action provenance. The corrected dataset keeps these values null/unknown rather than inferring them.
- The six historical `level_progress` labels remain explicitly unverified because their starting levels are unavailable.
- This is a preflight over the raw JSONL and NPY evidence plus in-memory dataset rows. No Parquet output was generated in this environment because `pyarrow` is not installed and package installation is unavailable here.
- The corrected builder writes to `data/parquet/ar25_experiment_qa_corrected.parquet`, preserving the existing Parquet artifact. Run the builder and validator in the project's `.venv` before accepting the corrected Parquet file.
- The validator's final CLI run, Parquet readback, and report generation remain pending on the user's Mac.
