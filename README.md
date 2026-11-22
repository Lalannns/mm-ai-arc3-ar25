### AR25 Experiment QA Dataset

**Dataset:** `data/parquet/ar25_experiment_qa.parquet`

| Metric | Result |
|---|---:|
| Valid unique QA pairs | **1,027** |
| Target | 1,000 |
| Shortfall | 0 |
| Validation errors | 0 |
| Parquet file size | 0.153 MB |
| Source exploration runs | 16 |
| Unique initial pixel states | 6 |
| Unique final pixel states | 106 |
| Unique visual transitions | 111 |

#### Outcome Breakdown

| Outcome | Records |
|---|---:|
| `changed` | 868 |
| `no_visible_change` | 142 |
| `changed_then_returned_to_start` | 11 |
| `level_progress` | 6 |
| **Total** | **1,027** |

#### Data Collection and Grounding

The dataset was generated from 16 exploration directories under `data/raw/exploration/`, including Level 0, Level 1, and Level 2 branching experiments and multi-step `ACTION1`–`ACTION7` sequences on `ar25-0c556536`.

Deduplication uses canonical 2D `int8` pixel-array SHA-256 hashes combined with action-sequence and question identity.

Answers use `answer_grounding: "observed_sequence_metadata"` and are grounded in recorded pixel differences, bounding boxes, color transitions, state transitions, and level progression.

#### Known Limitations

- Initial frames on `ar25-0c556536` were identical across tested seeds (`0`, `1`, and `2`).
- State diversity therefore comes from action sequences, coordinate-based interactions, and level/prefix progression.
- Different action sequences can converge on identical visual transitions.
- `estimate_counter_cells` approximates cell changes in the top strip rather than reading the environment's internal counter state directly.
- Unique QA records do not necessarily represent unique visual states or independent reasoning tasks.

#### Reproduce Build and Validation

Run from the repository root:

```zsh
.venv/bin/python scripts/16_build_experiment_qa_parquet.py && .venv/bin/python scripts/validate_experiment_qa.py
```

**Latest reported validation:** 1,027 records, 0 validation errors, target achieved.
