import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ar25_dataset_integrity import (  # noqa: E402
    canonical_array_hash,
    load_grid,
    trajectory_payload,
    trajectory_signature,
)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_module("ar25_builder", "16_build_experiment_qa_parquet.py")
validator = load_module("ar25_validator", "validate_experiment_qa.py")


def make_run(root, *, run="run_a", eid="exp_1", initial=None, after=None,
             actions=None, start=None, end=None, prefix=None,
             include_level_fields=True, changed_override=None):
    run_dir = root / run
    frames = run_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    initial = np.zeros((3, 3), dtype=np.int8) if initial is None else np.asarray(initial)
    after = [np.ones((3, 3), dtype=np.int8)] if after is None else [np.asarray(x) for x in after]
    actions = [{"action": "ACTION1", "action_data": {}}] if actions is None else actions
    np.save(frames / f"{eid}_step00_start.npy", initial)
    steps = []
    before = initial
    for i, (action, arr) in enumerate(zip(actions, after), start=1):
        filename = f"{eid}_step{i:02d}_after_00.npy"
        np.save(frames / filename, arr)
        changed = int(np.count_nonzero(before != arr)) if before.shape == arr.shape else int(arr.size)
        if changed_override is not None:
            changed = changed_override
        step = {
            "step": i, "action": action["action"], "action_data": action.get("action_data", {}),
            "after_files": [filename], "changed_cells": changed,
            "levels_completed": end if end is not None else 0,
        }
        steps.append(step)
        before = arr
    record = {
        "run_id": run, "experiment_id": eid, "question": "What happened?",
        "actions": actions, "steps_executed": len(steps), "steps": steps,
        "outcome": "changed", "final_state": "NOT_FINISHED",
    }
    if prefix is not None:
        record["prefix_actions"] = prefix
        record["prefix_actions_provenance"] = "observed"
    if include_level_fields:
        record.update({
            "levels_completed_start": start,
            "levels_completed_start_provenance": "observed" if start is not None else "unknown",
            "levels_completed_end": end,
            "levels_completed_end_provenance": "observed" if end is not None else "unknown",
        })
    (run_dir / "experiments.jsonl").write_text(json.dumps(record) + "\n")
    return run_dir


def test_identical_arrays_different_npy_files_have_same_hash(tmp_path):
    a = np.arange(9, dtype=np.int8).reshape(3, 3)
    b = np.asfortranarray(a)
    p1, p2 = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(p1, a); np.save(p2, b)
    assert canonical_array_hash(np.load(p1)) == canonical_array_hash(np.load(p2))


def test_different_pixel_arrays_have_different_hashes():
    a = np.zeros((3, 3), dtype=np.int8)
    b = a.copy(); b[0, 0] = 1
    assert canonical_array_hash(a) != canonical_array_hash(b)


def test_intermediate_trajectory_changes_signature_with_same_endpoints():
    base = dict(question="q", initial_hash="a" * 64, final_hash="c" * 64,
                outcome="changed", final_state="NOT_FINISHED", prefix_actions=[],
                levels_start=None, levels_end=0, levels_start_provenance="unknown",
                levels_end_provenance="observed")
    step_a = [{"action": "ACTION1", "action_data": {}, "after_frame_hashes": ["b" * 64, "c" * 64]}]
    step_b = [{"action": "ACTION1", "action_data": {}, "after_frame_hashes": ["d" * 64, "c" * 64]}]
    assert trajectory_signature(trajectory_payload(steps=step_a, **base)) != trajectory_signature(trajectory_payload(steps=step_b, **base))


def test_first_action_level_progression_preserves_start_and_end(tmp_path):
    make_run(tmp_path, start=0, end=1)
    rows, _ = builder.build_dataset(tmp_path)
    assert rows[0]["levels_completed_start"] == 0
    assert rows[0]["levels_completed_end"] == 1
    assert "levels_completed_start=0" in rows[0]["answer"]


def test_targeted_prefix_is_separate_from_target_actions(tmp_path):
    prefix = [{"action": "ACTION3", "action_data": {}}]
    target = [{"action": "ACTION2", "action_data": {"x": 1, "y": 2}}]
    make_run(tmp_path, start=1, end=1, prefix=prefix, actions=target)
    rows, _ = builder.build_dataset(tmp_path)
    row = rows[0]
    assert json.loads(row["prefix_actions"]) == prefix
    manifest = json.loads(row["trajectory_manifest"])
    assert [s["action"] for s in manifest] == ["ACTION2"]
    assert row["levels_completed_start"] == 1


def test_unknown_historical_start_remains_null_not_zero(tmp_path):
    make_run(tmp_path, start=None, end=1, include_level_fields=False)
    rows, _ = builder.build_dataset(tmp_path)
    assert rows[0]["levels_completed_start"] is None
    assert rows[0]["levels_completed_start_provenance"] == "unknown"
    assert "levels_completed_start=unknown" in rows[0]["answer"]


def test_missing_frame_is_rejected(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "frames" / "exp_1_step01_after_00.npy").unlink()
    with pytest.raises(FileNotFoundError):
        builder.build_dataset(tmp_path)


@pytest.mark.parametrize("bad", [np.zeros((3,), dtype=np.int8), np.zeros((2, 2, 2), dtype=np.int8)])
def test_invalid_frame_dimensions_are_rejected(tmp_path, bad):
    path = tmp_path / "bad.npy"; np.save(path, bad)
    with pytest.raises(ValueError):
        load_grid(path)


def test_inconsistent_action_step_count_is_rejected(tmp_path):
    run_dir = make_run(tmp_path)
    path = run_dir / "experiments.jsonl"
    rec = json.loads(path.read_text()); rec["actions"].append({"action": "ACTION2"})
    path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="target actions/steps mismatch"):
        builder.build_dataset(tmp_path)


def test_recorded_pixel_change_mismatch_is_rejected(tmp_path):
    make_run(tmp_path, changed_override=0)
    with pytest.raises(ValueError, match="changed_cells mismatch"):
        builder.build_dataset(tmp_path)


def test_duplicate_record_ids_are_detected():
    rows = [{"record_id": "same"}, {"record_id": "same"}, {"record_id": "other"}]
    assert validator.duplicate_values(rows, "record_id") == ["same"]


def test_invalid_json_schema_values_are_rejected():
    schema = json.loads((ROOT / "schemas" / "experiment_qa_schema.json").read_text())
    check = Draft202012Validator(schema)
    # The complete object is invalid specifically because the outcome is outside the enum.
    invalid = {
        "record_id": "x", "source_run": "r", "experiment_id": "e", "signature_hash": "0" * 64,
        "hash_version": "ar25-pixel-trajectory-v2", "question": "q", "answer": "a",
        "answer_grounding": "observed_raw_metadata", "step_count": 1,
        "action_sequence": "[]", "prefix_actions": "[]", "prefix_actions_provenance": "observed",
        "trajectory_manifest": "[]", "outcome": "bad", "final_state": "NOT_FINISHED",
        "levels_completed_start": None, "levels_completed_start_provenance": "unknown",
        "levels_completed_end": 0, "levels_completed_end_provenance": "observed",
        "total_changed_cells": 0,
        "total_changed_cells_semantics": "sum_of_per_step_full_frame_changed_cells; cells may be counted more than once",
        "initial_frame_hash": "0" * 64, "final_frame_hash": "0" * 64,
        "initial_frame_path": "a.npy", "intermediate_frame_paths": "[]",
        "final_frame_path": "b.npy", "frame_shape": "3x3",
    }
    messages = [e.message for e in check.iter_errors(invalid)]
    assert any("'bad' is not one of" in m for m in messages)


def test_incorrect_historical_return_label_is_rederived_from_saved_frames():
    initial = np.zeros((3, 3), dtype=np.int8)
    final = np.ones((3, 3), dtype=np.int8)
    outcome, provenance = builder._derive_outcome(
        "changed_then_returned_to_start", initial, final, total_changed=9,
        levels_start=None, levels_end=0,
    )
    assert outcome == "changed"
    assert provenance == "recomputed_from_pixel_sequence"


def test_historical_level_progress_without_start_is_explicitly_unverified():
    initial = np.zeros((3, 3), dtype=np.int8)
    final = np.ones((3, 3), dtype=np.int8)
    outcome, provenance = builder._derive_outcome(
        "level_progress", initial, final, total_changed=9,
        levels_start=None, levels_end=1,
    )
    assert outcome == "level_progress"
    assert provenance == "recorded_unverified"


def test_provenance_sidecar_recomputes_every_raw_source(tmp_path):
    make_run(tmp_path, start=0, end=0)
    rows, provenance = builder.build_dataset(tmp_path)
    sidecar = tmp_path / "provenance.json"
    sidecar.write_text(json.dumps(provenance))
    errors = []
    validator.validate_provenance(sidecar, rows, errors)
    assert errors == []
    assert provenance["raw_experiment_count"] == 1
    assert provenance["records"][0]["source_count"] == 1
