# AR25 Experiment QA Validation Report

- Status: **PASSED**
- QA rows: 1021
- Raw source experiments: 1300
- Collapsed source records mapped: 279
- Target: 1000 (shortfall: 0)
- Unknown starting levels: 1021
- Unknown prefix provenance: 1021
- Unverified historical level_progress labels: 6

## Outcome counts

- `changed`: 873
- `level_progress`: 6
- `no_visible_change`: 142

## Errors

- None

## Warnings

- row 110: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown
- row 204: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown
- row 298: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown
- row 490: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown
- row 510: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown
- row 943: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown
- historical provenance gaps: unknown starts=1021, unknown ends=0, unknown prefixes=1021; null/unknown values are preserved
- 6 level_progress labels are retained from raw metadata but cannot be independently reclassified without a starting level

## Research limitations

- QA rows are experiment/trajectory observations, not independent ARC reasoning problems.
- Repeated initial states and convergent trajectories create dependence; random row splits can leak related evidence.
- counter_cells_used is a heuristic estimate, not a verified internal game counter.
- This dataset documents observed experiment sequences; it does not prove autonomous VLM game solving.
- Answer checks verify declared numeric claims and provenance fields, not unrestricted natural-language semantic truth.
- Group train/validation/test splits by initial visual state and experiment family/trajectory to reduce leakage.
