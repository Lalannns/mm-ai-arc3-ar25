"""Independent structural and evidence validation for the AR25 experiment QA dataset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from ar25_dataset_integrity import (
    canonical_action,
    frame_hash,
    json_dumps,
    load_grid,
    shape_aware_changed_cells,
    trajectory_payload,
    trajectory_signature,
)

PARQUET_PATH = Path("data/parquet/ar25_experiment_qa_corrected.parquet")
SCHEMA_PATH = Path("schemas/experiment_qa_schema.json")
PROVENANCE_PATH = Path("data/parquet/ar25_experiment_qa_provenance.json")
TARGET_PAIRS = 1000


def _parse_json_field(row: dict[str, Any], key: str, expected_type: type, errors: list[str], row_no: int):
    try:
        value = json.loads(row[key])
        if not isinstance(value, expected_type):
            raise TypeError(f"expected {expected_type.__name__}, got {type(value).__name__}")
        return value
    except Exception as exc:
        errors.append(f"row {row_no}: invalid {key}: {exc}")
        return None


def validate_row(row: dict[str, Any], row_no: int, schema_validator: Draft202012Validator) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for err in schema_validator.iter_errors(row):
        path = ".".join(str(p) for p in err.absolute_path) or "$"
        errors.append(f"row {row_no}: JSON Schema {path}: {err.message}")

    actions = _parse_json_field(row, "action_sequence", list, errors, row_no)
    prefixes = _parse_json_field(row, "prefix_actions", list, errors, row_no)
    manifest = _parse_json_field(row, "trajectory_manifest", list, errors, row_no)
    intermediate = _parse_json_field(row, "intermediate_frame_paths", list, errors, row_no)
    if any(x is None for x in (actions, prefixes, manifest, intermediate)):
        return errors, warnings

    if len(actions) != row.get("step_count") or len(manifest) != row.get("step_count"):
        errors.append(f"row {row_no}: step_count/action_sequence/trajectory_manifest length mismatch")
        return errors, warnings
    if row.get("levels_completed_start") is None and row.get("levels_completed_start_provenance") != "unknown":
        errors.append(f"row {row_no}: null starting level must have unknown provenance")
    if row.get("levels_completed_end") is None and row.get("levels_completed_end_provenance") != "unknown":
        errors.append(f"row {row_no}: null ending level must have unknown provenance")
    if row.get("levels_completed_start") is not None and row.get("levels_completed_start_provenance") == "unknown":
        errors.append(f"row {row_no}: known starting level cannot have unknown provenance")
    if row.get("levels_completed_end") is not None and row.get("levels_completed_end_provenance") == "unknown":
        errors.append(f"row {row_no}: known ending level cannot have unknown provenance")
    if row.get("prefix_actions_provenance") == "unknown" and prefixes:
        errors.append(f"row {row_no}: prefix actions cannot be populated when prefix provenance is unknown")
    if row.get("prefix_actions_provenance") == "unknown" and row.get("answer_grounding") != "partially_unknown_raw_metadata":
        errors.append(f"row {row_no}: unknown prefix provenance requires partial answer grounding")
    expected_grounding = ("observed_raw_metadata" if row.get("levels_completed_start_provenance") != "unknown" and row.get("levels_completed_end_provenance") != "unknown" and row.get("prefix_actions_provenance") != "unknown" else "partially_unknown_raw_metadata")
    if row.get("answer_grounding") != expected_grounding:
        errors.append(f"row {row_no}: answer_grounding does not match field-level provenance")

    initial_path = Path(row["initial_frame_path"])
    final_path = Path(row["final_frame_path"])
    try:
        initial_arr = load_grid(initial_path)
        initial_hash = frame_hash(initial_path)
        if initial_hash != row["initial_frame_hash"]:
            errors.append(f"row {row_no}: initial_frame_hash does not match pixel content")
        expected_shape = f"{initial_arr.shape[0]}x{initial_arr.shape[1]}"
        if row["frame_shape"] != expected_shape:
            errors.append(f"row {row_no}: frame_shape mismatch; recorded={row['frame_shape']} actual={expected_shape}")
    except Exception as exc:
        errors.append(f"row {row_no}: invalid initial frame: {exc}")
        return errors, warnings

    previous_path = initial_path
    previous_hash = initial_hash
    cumulative = 0
    expected_intermediate: list[str] = []
    manifest_for_signature: list[dict[str, Any]] = []
    for step_index, step in enumerate(manifest):
        label = f"row {row_no} step {step_index + 1}"
        try:
            action = canonical_action(actions[step_index])
            if step.get("action") != action["action"] or (step.get("action_data") or {}) != action["action_data"]:
                errors.append(f"{label}: action does not match action_sequence")
            before_path = Path(step["before_frame_path"])
            before_hash = frame_hash(before_path)
            if before_hash != previous_hash:
                errors.append(f"{label}: before-frame does not chain from previous observation")
            if before_hash != step.get("before_frame_hash"):
                errors.append(f"{label}: recorded before_frame_hash mismatch")
            after_paths = step.get("after_frame_paths") or []
            after_hashes = step.get("after_frame_hashes") or []
            if not after_paths or len(after_paths) != len(after_hashes):
                errors.append(f"{label}: after frame paths/hashes are empty or inconsistent")
                continue
            computed_after_hashes = []
            for path_text, expected_hash in zip(after_paths, after_hashes):
                actual_hash = frame_hash(path_text)
                computed_after_hashes.append(actual_hash)
                if actual_hash != expected_hash:
                    errors.append(f"{label}: after-frame hash mismatch for {path_text}")
            before_arr = load_grid(before_path)
            after_path = Path(after_paths[-1])
            after_arr = load_grid(after_path)
            changed = shape_aware_changed_cells(before_arr, after_arr)
            if changed != step.get("changed_cells"):
                errors.append(f"{label}: changed_cells mismatch; recorded={step.get('changed_cells')} actual={changed}")
            cumulative += changed
            expected_intermediate.extend(after_paths[:-1])
            previous_path = after_path
            previous_hash = computed_after_hashes[-1]
            manifest_for_signature.append({
                "action": action["action"],
                "action_data": action["action_data"],
                "after_frame_hashes": computed_after_hashes,
            })
        except Exception as exc:
            errors.append(f"{label}: evidence validation failed: {exc}")

    level_after_values = [step.get("levels_completed_after") for step in manifest]
    level_before_values = [step.get("levels_completed_before") for step in manifest]
    if row.get("levels_completed_start") is not None and level_before_values[0] is not None and row["levels_completed_start"] != level_before_values[0]:
        errors.append(f"row {row_no}: first target-step level does not match recorded starting level")
    for i in range(1, len(manifest)):
        if level_after_values[i - 1] is not None and level_before_values[i] is not None and level_after_values[i - 1] != level_before_values[i]:
            errors.append(f"row {row_no}: level metadata does not chain between steps {i} and {i + 1}")
    if row.get("levels_completed_end") is not None and level_after_values[-1] is not None and row["levels_completed_end"] != level_after_values[-1]:
        errors.append(f"row {row_no}: final step level does not match recorded ending level")
    if expected_intermediate != intermediate:
        errors.append(f"row {row_no}: intermediate_frame_paths does not match trajectory_manifest")
    try:
        if frame_hash(final_path) != row["final_frame_hash"]:
            errors.append(f"row {row_no}: final_frame_hash does not match pixel content")
        if frame_hash(final_path) != previous_hash:
            errors.append(f"row {row_no}: final frame is not the last trajectory observation")
    except Exception as exc:
        errors.append(f"row {row_no}: invalid final frame: {exc}")
    if cumulative != row.get("total_changed_cells"):
        errors.append(f"row {row_no}: total_changed_cells mismatch; recorded={row.get('total_changed_cells')} actual={cumulative}")

    try:
        final_arr = load_grid(final_path)
        any_change = cumulative > 0
        returned_to_start = initial_arr.shape == final_arr.shape and np.array_equal(initial_arr, final_arr)
        start_level, end_level = row.get("levels_completed_start"), row.get("levels_completed_end")
        if start_level is not None and end_level is not None and end_level > start_level:
            expected_outcome = "level_progress"
            expected_provenance = "recomputed_from_observed_levels"
        elif start_level is None and row.get("outcome") == "level_progress":
            expected_outcome = row.get("recorded_outcome")
            expected_provenance = "recorded_unverified"
            warnings.append(f"row {row_no}: level_progress is retained from raw metadata but cannot be independently verified because starting level is unknown")
        elif not any_change:
            expected_outcome = "no_visible_change"
            expected_provenance = "recomputed_from_pixel_sequence"
        elif returned_to_start:
            expected_outcome = "changed_then_returned_to_start"
            expected_provenance = "recomputed_from_pixel_sequence"
        else:
            expected_outcome = "changed"
            expected_provenance = "recomputed_from_pixel_sequence"
        if row.get("outcome") != expected_outcome or row.get("outcome_provenance") != expected_provenance:
            errors.append(f"row {row_no}: outcome/provenance mismatch; recorded={row.get('outcome')}/{row.get('outcome_provenance')} recomputed={expected_outcome}/{expected_provenance}")
    except Exception as exc:
        errors.append(f"row {row_no}: outcome evidence check failed: {exc}")

    try:
        payload = trajectory_payload(
            question=row["question"], initial_hash=initial_hash,
            steps=manifest_for_signature, final_hash=previous_hash,
            outcome=row.get("outcome"), final_state=row.get("final_state"),
            prefix_actions=prefixes, levels_start=row.get("levels_completed_start"),
            levels_end=row.get("levels_completed_end"),
            levels_start_provenance=row["levels_completed_start_provenance"],
            levels_end_provenance=row["levels_completed_end_provenance"],
        )
        if trajectory_signature(payload) != row.get("signature_hash"):
            errors.append(f"row {row_no}: signature_hash does not match recomputed trajectory identity")
    except Exception as exc:
        errors.append(f"row {row_no}: cannot recompute trajectory signature: {exc}")

    # Check the builder's explicit factual claims, without pretending to prove natural-language semantics.
    answer = row.get("answer", "")
    prefix_provenance = row.get("prefix_actions_provenance")
    if prefix_provenance == "observed":
        prefix_claim = f"prefix actions: observed ({len(prefixes)} actions)" if prefixes else "prefix actions: observed empty prefix"
    elif prefix_provenance == "reconstructed":
        prefix_claim = f"prefix actions: reconstructed ({len(prefixes)} actions)"
    else:
        prefix_claim = "prefix actions: unknown (not stored in this historical record)"
    claims = [
        f"Evidence-derived outcome: {row.get('outcome')}",
        f"cumulative per-step changed_cells={cumulative}",
        f"levels_completed_start={'unknown' if row.get('levels_completed_start') is None else row['levels_completed_start']}",
        f"levels_completed_end={'unknown' if row.get('levels_completed_end') is None else row['levels_completed_end']}",
        prefix_claim,
    ]
    if row.get("recorded_outcome") != row.get("outcome"):
        claims.append(f"Raw recorded outcome label={row.get('recorded_outcome')}")
    for claim in claims:
        if claim not in answer:
            errors.append(f"row {row_no}: answer omits or contradicts evidence claim {claim!r}")
    return errors, warnings


def duplicate_values(rows: list[dict[str, Any]], key: str) -> list[Any]:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for row in rows:
        value = row.get(key)
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validate_provenance(path: Path, rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing provenance sidecar: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid provenance sidecar JSON: {exc}")
        return {}
    row_by_id = {r.get("record_id"): r for r in rows}
    mapped_ids = [r.get("record_id") for r in data.get("records", [])]
    if set(mapped_ids) != set(row_by_id) or len(mapped_ids) != len(set(mapped_ids)):
        errors.append("provenance record IDs do not map one-to-one to Parquet record IDs")
    mapped_sources = sum(int(r.get("source_count", 0)) for r in data.get("records", []))
    if mapped_sources != data.get("raw_experiment_count"):
        errors.append(f"provenance source count mismatch: mapped={mapped_sources}, declared={data.get('raw_experiment_count')}")
    if data.get("deduplicated_record_count") != len(rows):
        errors.append("provenance deduplicated_record_count does not match Parquet row count")

    seen_sources: set[tuple[str, int]] = set()
    raw_cache: dict[str, list[str]] = {}
    for canonical in data.get("records", []):
        canonical_id = canonical.get("record_id")
        row = row_by_id.get(canonical_id)
        if row is None or row.get("signature_hash") != canonical.get("signature_hash"):
            errors.append(f"provenance signature/record mismatch for {canonical_id}")
            continue
        actual_source_count = len(canonical.get("sources", []))
        if actual_source_count != canonical.get("source_count"):
            errors.append(f"provenance source_count mismatch for {canonical_id}")

        for source in canonical.get("sources", []):
            source_file = str(source.get("source_file", ""))
            raw_index = source.get("raw_record_index")
            identity = (source_file, raw_index)
            if identity in seen_sources:
                errors.append(f"duplicate raw source mapping: {identity}")
            seen_sources.add(identity)
            try:
                if source_file not in raw_cache:
                    raw_cache[source_file] = Path(source_file).read_text(encoding="utf-8").splitlines()
                lines = raw_cache[source_file]
                if not isinstance(raw_index, int) or raw_index < 0 or raw_index >= len(lines):
                    raise IndexError(f"raw_record_index out of range: {raw_index}")
                raw = json.loads(lines[raw_index])
                if raw.get("experiment_id") != source.get("experiment_id") or raw.get("run_id") != source.get("run_id"):
                    raise ValueError("raw record identity does not match provenance mapping")
                if str(raw.get("outcome", "unknown")) != source.get("recorded_outcome"):
                    raise ValueError("recorded_outcome does not match raw JSONL")
                steps = raw.get("steps") or []
                if not steps or (raw.get("steps_executed") is not None and int(raw["steps_executed"]) != len(steps)):
                    raise ValueError("raw steps_executed is missing/inconsistent")
                if raw.get("actions") is not None and len(raw["actions"]) != len(steps):
                    raise ValueError("raw target action count does not match steps")

                run_dir = Path(source_file).parent
                initial_path = run_dir / "frames" / f"{raw['experiment_id']}_step00_start.npy"
                initial_arr = load_grid(initial_path)
                initial_hash = frame_hash(initial_path)
                previous_path, previous_arr = initial_path, initial_arr
                signature_steps = []
                total_changed = 0
                for step_index, step in enumerate(steps):
                    after_files = step.get("after_files") or []
                    if not after_files:
                        raise ValueError(f"step {step_index + 1} has no after_files")
                    before_path = run_dir / "frames" / step["before_file"] if step.get("before_file") else previous_path
                    before_arr = load_grid(before_path)
                    if frame_hash(before_path) != frame_hash(previous_path):
                        raise ValueError(f"step {step_index + 1} before-frame chain mismatch")
                    after_paths = [run_dir / "frames" / name for name in after_files]
                    after_hashes = [frame_hash(p) for p in after_paths]
                    after_arr = load_grid(after_paths[-1])
                    changed = shape_aware_changed_cells(before_arr, after_arr)
                    if step.get("changed_cells") is not None and int(step["changed_cells"]) != changed:
                        raise ValueError(f"step {step_index + 1} changed_cells mismatch")
                    total_changed += changed
                    signature_steps.append({
                        "action": step["action"],
                        "action_data": step.get("action_data") or {},
                        "after_frame_hashes": after_hashes,
                    })
                    previous_path, previous_arr = after_paths[-1], after_arr

                if raw.get("levels_completed_start") is not None:
                    levels_start = int(raw["levels_completed_start"])
                    start_prov = raw.get("levels_completed_start_provenance", "observed")
                else:
                    levels_start, start_prov = None, "unknown"
                if raw.get("levels_completed_end") is not None:
                    levels_end = int(raw["levels_completed_end"])
                    end_prov = raw.get("levels_completed_end_provenance", "observed")
                elif steps[-1].get("levels_completed") is not None:
                    levels_end, end_prov = int(steps[-1]["levels_completed"]), "observed"
                else:
                    levels_end, end_prov = None, "unknown"
                raw_outcome = str(raw.get("outcome", "unknown"))
                if levels_start is not None and levels_end is not None and levels_end > levels_start:
                    derived_outcome = "level_progress"
                elif levels_start is None and raw_outcome == "level_progress":
                    derived_outcome = "level_progress"
                elif total_changed == 0:
                    derived_outcome = "no_visible_change"
                elif initial_arr.shape == previous_arr.shape and np.array_equal(initial_arr, previous_arr):
                    derived_outcome = "changed_then_returned_to_start"
                else:
                    derived_outcome = "changed"
                prefix_actions = raw.get("prefix_actions") or []
                payload = trajectory_payload(
                    question=str(raw["question"]), initial_hash=initial_hash, steps=signature_steps,
                    final_hash=frame_hash(previous_path), outcome=derived_outcome,
                    final_state=raw.get("final_state"), prefix_actions=prefix_actions,
                    levels_start=levels_start, levels_end=levels_end,
                    levels_start_provenance=start_prov, levels_end_provenance=end_prov,
                )
                source_signature = trajectory_signature(payload)
                if source_signature != canonical.get("signature_hash"):
                    raise ValueError("raw source recomputed trajectory signature does not match canonical mapping")
            except Exception as exc:
                errors.append(f"invalid provenance source {identity}: {exc}")

    if len(seen_sources) != data.get("raw_experiment_count"):
        errors.append(f"provenance unique raw-source count mismatch: {len(seen_sources)}")
    return data



def validate_unsuccessful_summaries(
    root: Path,
    errors: list[str],
    expected_main_count: int | None = None,
) -> dict[str, Any]:
    """Compare unsuccessful summary records with their main experiment records."""
    main_records = {}
    summary_count = 0
    main_count = 0
    verified_count = 0
    missing_count = 0
    mismatch_count = 0
    malformed_count = 0
    duplicate_count = 0

    fields = ("question", "actions", "outcome", "error", "final_state")

    def read_jsonl(file_path):
        nonlocal malformed_count
        with file_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not a JSON object")
                    yield line_no, record
                except (json.JSONDecodeError, ValueError) as exc:
                    malformed_count += 1
                    errors.append(f"Invalid JSONL {file_path}:{line_no}: {exc}")

    main_files = sorted(root.rglob("experiments.jsonl"))
    summary_files = sorted(root.rglob("unsuccessful_experiments.jsonl"))

    for file_path in main_files:
        for line_no, record in read_jsonl(file_path):
            main_count += 1
            identity = (record.get("run_id"), record.get("experiment_id"))
            if not all(isinstance(x, str) and x for x in identity):
                errors.append(f"Invalid main-record identity: {file_path}:{line_no}")
                continue
            if identity in main_records:
                duplicate_count += 1
                errors.append(f"Duplicate main-record identity: {identity}")
                continue
            main_records[identity] = record

    seen_summaries = set()

    for file_path in summary_files:
        for line_no, summary in read_jsonl(file_path):
            summary_count += 1
            identity = (summary.get("run_id"), summary.get("experiment_id"))

            if not all(isinstance(x, str) and x for x in identity):
                missing_count += 1
                errors.append(f"Invalid summary identity: {file_path}:{line_no}")
                continue

            if identity in seen_summaries:
                duplicate_count += 1
                errors.append(f"Duplicate summary identity: {identity}")
                continue
            seen_summaries.add(identity)

            main = main_records.get(identity)
            if main is None:
                missing_count += 1
                errors.append(f"Summary has no matching main record: {identity}")
                continue

            record_ok = True
            for field in fields:
                if field not in summary or field not in main or summary[field] != main[field]:
                    mismatch_count += 1
                    record_ok = False
                    errors.append(f"Summary mismatch for {identity}: {field}")

            if record_ok:
                verified_count += 1

    if expected_main_count is not None and main_count != expected_main_count:
        errors.append(
            f"Main source count mismatch: {main_count} != provenance {expected_main_count}"
        )

    if malformed_count:
        errors.append(f"Malformed JSONL records: {malformed_count}")

    return {
        "main_experiment_count": main_count,
        "unsuccessful_summary_count": summary_count,
        "physical_jsonl_entry_count": main_count + summary_count,
        "verified_duplicate_summary_count": verified_count,
        "missing_summary_matches": missing_count,
        "field_mismatches": mismatch_count,
        "duplicate_identities": duplicate_count,
        "malformed_jsonl_records": malformed_count,
        "main_files_found": len(main_files),
        "summary_files_found": len(summary_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PARQUET_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    parser.add_argument("--json-report", type=Path, default=Path("reports/experiment_qa_validation_report.json"))
    parser.add_argument("--md-report", type=Path, default=Path("reports/experiment_qa_validation_report.md"))
    parser.add_argument("--target", type=int, default=TARGET_PAIRS)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Missing Parquet file: {args.input}")
    import pyarrow.parquet as pq

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    schema_validator = Draft202012Validator(schema)
    rows = pq.read_table(args.input).to_pylist()
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_ids = duplicate_values(rows, "record_id")
    duplicate_sigs = duplicate_values(rows, "signature_hash")
    for value in duplicate_ids:
        errors.append(f"duplicate record_id: {value}")
    for value in duplicate_sigs:
        errors.append(f"duplicate signature_hash: {value}")
    seen_ids = {r.get("record_id") for r in rows}
    seen_sigs = {r.get("signature_hash") for r in rows}
    for row_no, row in enumerate(rows):
        row_errors, row_warnings = validate_row(row, row_no, schema_validator)
        errors.extend(row_errors)
        warnings.extend(row_warnings)

    provenance = validate_provenance(args.provenance, rows, errors)
    source_inventory = validate_unsuccessful_summaries(
        Path("data/raw/exploration"),
        errors,
        expected_main_count=provenance.get("raw_experiment_count"),
    )
    unknown_start_count = sum(r.get("levels_completed_start") is None for r in rows)
    unknown_end_count = sum(r.get("levels_completed_end") is None for r in rows)
    unknown_prefix_count = sum(r.get("prefix_actions_provenance") == "unknown" for r in rows)
    unverified_outcome_count = sum(r.get("outcome_provenance") == "recorded_unverified" for r in rows)
    if unknown_start_count or unknown_end_count or unknown_prefix_count:
        warnings.append(f"historical provenance gaps: unknown starts={unknown_start_count}, unknown ends={unknown_end_count}, unknown prefixes={unknown_prefix_count}; null/unknown values are preserved")
    if unverified_outcome_count:
        warnings.append(f"{unverified_outcome_count} level_progress labels are retained from raw metadata but cannot be independently reclassified without a starting level")
    outcomes = Counter(r.get("outcome") for r in rows)
    report = {
        "status": "passed" if not errors else "failed",
        "parquet_path": str(args.input),
        "schema_path": str(args.schema),
        "rows": len(rows),
        "unique_record_ids": len(seen_ids),
        "unique_signatures": len(seen_sigs),
        "raw_experiments": provenance.get("raw_experiment_count"),
        "collapsed_source_records": provenance.get("collapsed_source_count"),
        "source_inventory": source_inventory,
        "target_pairs": args.target,
        "target_shortfall": max(0, args.target - len(rows)),
        "outcomes": dict(outcomes),
        "unknown_start_level_rows": unknown_start_count,
        "unknown_end_level_rows": unknown_end_count,
        "unknown_prefix_rows": unknown_prefix_count,
        "unverified_outcome_rows": unverified_outcome_count,
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "QA rows are experiment/trajectory observations, not independent ARC reasoning problems.",
            "Repeated initial states and convergent trajectories create dependence; random row splits can leak related evidence.",
            "counter_cells_used is a heuristic estimate, not a verified internal game counter.",
            "This dataset documents observed experiment sequences; it does not prove autonomous VLM game solving.",
            "Answer checks verify declared numeric claims and provenance fields, not unrestricted natural-language semantic truth.",
            "Group train/validation/test splits by initial visual state and experiment family/trajectory to reduce leakage.",
        ],
    }
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.md_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# AR25 Experiment QA Validation Report", "",
        f"- Status: **{report['status'].upper()}**",
        f"- QA rows: {len(rows)}",
        f"- Raw source experiments: {report['raw_experiments']}",
        f"- Collapsed source records mapped: {report['collapsed_source_records']}",
        f"- Target: {args.target} (shortfall: {report['target_shortfall']})",
        f"- Unknown starting levels: {unknown_start_count}",
        f"- Unknown prefix provenance: {unknown_prefix_count}",
        f"- Unverified historical level_progress labels: {unverified_outcome_count}", "",
        "## Outcome counts", "",
    ]
    md.extend(f"- `{k}`: {v}" for k, v in sorted(outcomes.items()))
    md += ["", "## Errors", ""] + ([f"- {e}" for e in errors] if errors else ["- None"])
    md += ["", "## Warnings", ""] + ([f"- {w}" for w in warnings] if warnings else ["- None"])
    md += ["", "## Research limitations", ""] + [f"- {x}" for x in report["limitations"]]
    args.md_report.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Status: {report['status']}")
    print(f"Rows: {len(rows)} | raw sources: {report['raw_experiments']} | mapped collapsed: {report['collapsed_source_records']}")
    print(f"Errors: {len(errors)} | warnings: {len(warnings)} | target shortfall: {report['target_shortfall']}")
    print(f"JSON report: {args.json_report}")
    print(f"Markdown report: {args.md_report}")
    if errors:
        for error in errors[:30]:
            print("ERROR:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
