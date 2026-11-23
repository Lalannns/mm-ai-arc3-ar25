"""Build the AR25 experiment QA dataset without mutating raw exploration evidence."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ar25_dataset_integrity import (
    canonical_action,
    frame_hash,
    json_dumps,
    load_grid,
    shape_aware_changed_cells,
    trajectory_payload,
    trajectory_signature,
)

BASE_DIR = Path("data/raw/exploration")
OUT_PATH = Path("data/parquet/ar25_experiment_qa_corrected.parquet")
PROVENANCE_PATH = Path("data/parquet/ar25_experiment_qa_provenance.json")


def iter_run_folders(base_dir: Path) -> list[Path]:
    if not base_dir.is_dir():
        return []
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and (p / "experiments.jsonl").is_file())


def _relative_frame(run_dir: Path, filename: str) -> tuple[Path, str]:
    path = run_dir / "frames" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame: {path}")
    return path, path.as_posix()


def _level_metadata(exp: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[int | None, int | None, str, str]:
    if exp.get("levels_completed_start") is not None:
        start = int(exp["levels_completed_start"])
        start_provenance = exp.get("levels_completed_start_provenance", "observed")
    else:
        start = None
        start_provenance = "unknown"

    if exp.get("levels_completed_end") is not None:
        end = int(exp["levels_completed_end"])
        end_provenance = exp.get("levels_completed_end_provenance", "observed")
    elif steps and steps[-1].get("levels_completed") is not None:
        end = int(steps[-1]["levels_completed"])
        end_provenance = "observed"
    else:
        end = None
        end_provenance = "unknown"
    return start, end, start_provenance, end_provenance


def _answer(
    question: str,
    steps: list[dict[str, Any]],
    outcome: Any,
    final_state: Any,
    total_changed: int,
    levels_start: int | None,
    levels_end: int | None,
    start_provenance: str,
    end_provenance: str,
    prefix_actions: list[Any],
    prefix_provenance: str,
) -> tuple[str, str]:
    actions = [canonical_action(s) for s in steps]
    actions_text = " -> ".join(
        f"{a['action']}({json_dumps(a['action_data'])})" if a["action_data"] else a["action"]
        for a in actions
    )
    start_text = str(levels_start) if levels_start is not None else "unknown"
    end_text = str(levels_end) if levels_end is not None else "unknown"
    if prefix_provenance == "observed":
        prefix_text = f"observed ({len(prefix_actions)} actions)" if prefix_actions else "observed empty prefix"
    elif prefix_provenance == "reconstructed":
        prefix_text = f"reconstructed ({len(prefix_actions)} actions)"
    else:
        prefix_text = "unknown (not stored in this historical record)"
    answer = (
        f"Question: {question} Executed target actions: {actions_text} ({len(steps)} steps). "
        f"Evidence-derived outcome: {outcome}; final_state={final_state}; "
        f"cumulative per-step changed_cells={total_changed}. "
        f"levels_completed_start={start_text} ({start_provenance}); "
        f"levels_completed_end={end_text} ({end_provenance}); prefix actions: {prefix_text}."
    )
    grounding = ("observed_raw_metadata" if start_provenance != "unknown" and end_provenance != "unknown" and prefix_provenance != "unknown" else "partially_unknown_raw_metadata")
    return answer, grounding


def _derive_outcome(raw_outcome: str, initial_arr: np.ndarray, final_arr: np.ndarray,
                    total_changed: int, levels_start: int | None, levels_end: int | None) -> tuple[str, str]:
    if levels_start is not None and levels_end is not None and levels_end > levels_start:
        return "level_progress", "recomputed_from_observed_levels"
    if levels_start is None and raw_outcome == "level_progress":
        # Preserve this historical label, but explicitly mark it unverified because the start level is absent.
        return "level_progress", "recorded_unverified"
    if total_changed == 0:
        return "no_visible_change", "recomputed_from_pixel_sequence"
    if initial_arr.shape == final_arr.shape and np.array_equal(initial_arr, final_arr):
        return "changed_then_returned_to_start", "recomputed_from_pixel_sequence"
    return "changed", "recomputed_from_pixel_sequence"


def build_dataset(base_dir: Path = BASE_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_signature: dict[str, dict[str, Any]] = {}
    sources_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_count = 0

    for run_dir in iter_run_folders(base_dir):
        exp_file = run_dir / "experiments.jsonl"
        for raw_index, line in enumerate(exp_file.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            exp = json.loads(line)
            raw_count += 1
            eid = str(exp["experiment_id"])
            steps = exp.get("steps") or []
            if not steps:
                raise ValueError(f"Experiment {eid} in {run_dir.name} has no steps")
            if exp.get("steps_executed") is not None and int(exp["steps_executed"]) != len(steps):
                raise ValueError(f"steps_executed mismatch in {run_dir.name}/{eid}")
            if exp.get("actions") is not None and len(exp["actions"]) != len(steps):
                raise ValueError(f"target actions/steps mismatch in {run_dir.name}/{eid}")

            init_path, init_rel = _relative_frame(run_dir, f"{eid}_step00_start.npy")
            init_arr = load_grid(init_path)
            initial_hash = frame_hash(init_path)
            prefix_actions = exp.get("prefix_actions") or []
            prefix_provenance = exp.get("prefix_actions_provenance", "unknown")
            levels_start, levels_end, start_prov, end_prov = _level_metadata(exp, steps)

            manifest_steps: list[dict[str, Any]] = []
            intermediate_paths: list[str] = []
            total_changed = 0
            previous_path = init_path
            previous_arr = init_arr
            for step_index, step in enumerate(steps):
                after_files = step.get("after_files") or []
                if not after_files:
                    raise ValueError(f"No after_files in {run_dir.name}/{eid} step {step_index + 1}")
                before_file = step.get("before_file")
                if before_file:
                    before_path, before_rel = _relative_frame(run_dir, before_file)
                    before_arr = load_grid(before_path)
                    if frame_hash(before_path) != frame_hash(previous_path):
                        raise ValueError(f"Frame chain mismatch before {run_dir.name}/{eid} step {step_index + 1}")
                else:
                    before_path, before_rel, before_arr = previous_path, previous_path.as_posix(), previous_arr

                after_paths: list[str] = []
                after_hashes: list[str] = []
                for filename in after_files:
                    after_path, after_rel = _relative_frame(run_dir, filename)
                    after_paths.append(after_rel)
                    after_hashes.append(frame_hash(after_path))
                after_path = Path(after_paths[-1])
                after_arr = load_grid(after_path)
                recomputed_changed = shape_aware_changed_cells(before_arr, after_arr)
                recorded_changed = step.get("changed_cells")
                if recorded_changed is not None and int(recorded_changed) != recomputed_changed:
                    raise ValueError(
                        f"changed_cells mismatch in {run_dir.name}/{eid} step {step_index + 1}: "
                        f"recorded={recorded_changed}, recomputed={recomputed_changed}"
                    )
                total_changed += recomputed_changed
                manifest_steps.append({
                    "step": int(step.get("step", step_index + 1)),
                    "action": canonical_action({"action": step["action"], "action_data": step.get("action_data") or {}})["action"],
                    "action_data": step.get("action_data") or {},
                    "before_frame_path": before_rel,
                    "before_frame_hash": frame_hash(before_path),
                    "after_frame_paths": after_paths,
                    "after_frame_hashes": after_hashes,
                    "changed_cells": recomputed_changed,
                    "levels_completed_before": step.get("levels_completed_before"),
                    "levels_completed_after": step.get("levels_completed_after", step.get("levels_completed")),
                })
                intermediate_paths.extend(after_paths[:-1])
                previous_path, previous_arr = after_path, after_arr

            final_path = previous_path
            final_arr = previous_arr
            final_hash = frame_hash(final_path)
            recorded_outcome = str(exp.get("outcome", "unknown"))
            outcome, outcome_provenance = _derive_outcome(
                recorded_outcome, init_arr, final_arr, total_changed, levels_start, levels_end
            )
            payload = trajectory_payload(
                question=str(exp["question"]), initial_hash=initial_hash, steps=manifest_steps,
                final_hash=final_hash, outcome=outcome, final_state=exp.get("final_state"),
                prefix_actions=prefix_actions, levels_start=levels_start, levels_end=levels_end,
                levels_start_provenance=start_prov, levels_end_provenance=end_prov,
            )
            sig_hash = trajectory_signature(payload)
            source = {
                "source_file": (exp_file).as_posix(),
                "source_run": run_dir.name,
                "raw_record_index": raw_index,
                "experiment_id": eid,
                "run_id": exp.get("run_id"),
                "seed": exp.get("seed"),
                "timestamp_utc": exp.get("timestamp_utc"),
                "prefix_actions": prefix_actions,
                "prefix_actions_provenance": prefix_provenance,
                "levels_completed_start": levels_start,
                "levels_completed_start_provenance": start_prov,
                "levels_completed_end": levels_end,
                "levels_completed_end_provenance": end_prov,
                "recorded_outcome": recorded_outcome,
                "derived_outcome": outcome,
                "outcome_provenance": outcome_provenance,
            }
            sources_by_signature[sig_hash].append(source)
            if sig_hash in rows_by_signature:
                continue

            action_objects = [canonical_action(s) for s in steps]
            answer, grounding = _answer(
                str(exp["question"]), steps, outcome, exp.get("final_state"),
                total_changed, levels_start, levels_end, start_prov, end_prov, prefix_actions, prefix_provenance,
            )
            if recorded_outcome != outcome:
                answer += f" Raw recorded outcome label={recorded_outcome}; derived label is retained with provenance={outcome_provenance}."
            record_id = f"ar25_{sig_hash[:20]}"
            rows_by_signature[sig_hash] = {
                "record_id": record_id,
                "source_run": run_dir.name,
                "experiment_id": eid,
                "signature_hash": sig_hash,
                "hash_version": payload["hash_version"],
                "question": str(exp["question"]),
                "answer": answer,
                "answer_grounding": grounding,
                "step_count": len(steps),
                "action_sequence": json_dumps(action_objects),
                "prefix_actions": json_dumps(prefix_actions),
                "prefix_actions_provenance": prefix_provenance,
                "trajectory_manifest": json_dumps(manifest_steps),
                "outcome": outcome,
                "recorded_outcome": recorded_outcome,
                "outcome_provenance": outcome_provenance,
                "final_state": exp.get("final_state"),
                "levels_completed_start": levels_start,
                "levels_completed_start_provenance": start_prov,
                "levels_completed_end": levels_end,
                "levels_completed_end_provenance": end_prov,
                "total_changed_cells": int(total_changed),
                "total_changed_cells_semantics": "sum_of_per_step_full_frame_changed_cells; cells may be counted more than once",
                "initial_frame_hash": initial_hash,
                "final_frame_hash": final_hash,
                "initial_frame_path": init_rel,
                "intermediate_frame_paths": json_dumps(intermediate_paths),
                "final_frame_path": final_path.as_posix(),
                "frame_shape": f"{init_arr.shape[0]}x{init_arr.shape[1]}",
            }

    rows = [rows_by_signature[sig] for sig in sorted(rows_by_signature)]
    provenance = {
        "format_version": 1,
        "hash_version": "ar25-pixel-trajectory-v2",
        "raw_experiment_count": raw_count,
        "deduplicated_record_count": len(rows),
        "collapsed_source_count": raw_count - len(rows),
        "deduplication_rule": "same question, canonical initial/final pixel hashes, ordered target actions and all ordered after-frame hashes, outcome/final state, prefix actions, and level values with provenance",
        "records": [
            {"record_id": rows_by_signature[sig]["record_id"], "signature_hash": sig, "source_count": len(sources_by_signature[sig]), "sources": sources_by_signature[sig]}
            for sig in sorted(rows_by_signature)
        ],
    }
    return rows, provenance



def build_arrow_schema(pa):
    """Pin nullable level columns to int64 even when this snapshot is all-null."""
    string_fields = [
        "record_id", "source_run", "experiment_id", "signature_hash", "hash_version",
        "question", "answer", "answer_grounding", "action_sequence", "prefix_actions",
        "prefix_actions_provenance", "trajectory_manifest", "outcome", "recorded_outcome",
        "outcome_provenance", "levels_completed_start_provenance",
        "levels_completed_end_provenance", "total_changed_cells_semantics",
        "initial_frame_hash", "final_frame_hash", "initial_frame_path",
        "intermediate_frame_paths", "final_frame_path", "frame_shape",
    ]
    fields = [pa.field(name, pa.string(), nullable=name == "final_state") for name in string_fields]
    fields.append(pa.field("final_state", pa.string(), nullable=True))
    fields.append(pa.field("step_count", pa.int64(), nullable=False))
    fields.append(pa.field("levels_completed_start", pa.int64(), nullable=True))
    fields.append(pa.field("levels_completed_end", pa.int64(), nullable=True))
    fields.append(pa.field("total_changed_cells", pa.int64(), nullable=False))
    return pa.schema(fields)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--provenance-output", type=Path, default=PROVENANCE_PATH)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, provenance = build_dataset(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=build_arrow_schema(pa))
    pq.write_table(table, args.output)
    args.provenance_output.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Raw experiments: {provenance['raw_experiment_count']}")
    print(f"Unique trajectory QA records: {len(rows)}")
    print(f"Collapsed source records retained in provenance: {provenance['collapsed_source_count']}")
    print(f"Parquet: {args.output}")
    print(f"Provenance: {args.provenance_output}")


if __name__ == "__main__":
    main()
