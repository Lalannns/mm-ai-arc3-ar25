"""Shared deterministic integrity primitives for the AR25 experiment QA dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HASH_VERSION = "ar25-pixel-trajectory-v2"
SUPPORTED_DTYPE_KINDS = {"b", "i", "u"}


def load_grid(path: str | Path) -> np.ndarray:
    arr = np.load(Path(path), allow_pickle=False)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError(f"Expected a 2D grid at {path}; got {getattr(arr, 'shape', None)}")
    if arr.dtype.kind not in SUPPORTED_DTYPE_KINDS:
        raise ValueError(f"Unsupported grid dtype at {path}: {arr.dtype}")
    if any(int(dim) <= 0 for dim in arr.shape):
        raise ValueError(f"Grid has an empty dimension at {path}: {arr.shape}")
    return arr


def canonical_array_hash(arr: np.ndarray) -> str:
    """Hash dtype, shape and C-order pixel bytes; independent of NPY headers/layout."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError("Canonical frame hashing requires a 2D numpy array")
    if arr.dtype.kind not in SUPPORTED_DTYPE_KINDS:
        raise ValueError(f"Unsupported frame dtype: {arr.dtype}")
    contiguous = np.ascontiguousarray(arr)
    header = json.dumps(
        {"version": HASH_VERSION, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"AR25-FRAME\x00")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def frame_hash(path: str | Path) -> str:
    return canonical_array_hash(load_grid(path))


def canonical_action(action: Any) -> dict[str, Any]:
    if isinstance(action, str):
        return {"action": action, "action_data": {}}
    if not isinstance(action, dict):
        raise ValueError(f"Invalid action specification: {action!r}")
    name = action.get("action")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Action has no valid action name: {action!r}")
    return {"action": name, "action_data": action.get("action_data") or {}}


def trajectory_payload(
    *,
    question: str,
    initial_hash: str,
    steps: list[dict[str, Any]],
    final_hash: str,
    outcome: Any,
    final_state: Any,
    prefix_actions: list[Any],
    levels_start: int | None,
    levels_end: int | None,
    levels_start_provenance: str,
    levels_end_provenance: str,
) -> dict[str, Any]:
    step_payload = []
    for step in steps:
        step_payload.append({
            "action": canonical_action({"action": step["action"], "action_data": step.get("action_data") or {}}),
            "after_frame_hashes": list(step["after_frame_hashes"]),
        })
    return {
        "hash_version": HASH_VERSION,
        "question": question,
        "initial_frame_hash": initial_hash,
        "steps": step_payload,
        "final_frame_hash": final_hash,
        "outcome": outcome,
        "final_state": final_state,
        "prefix_actions": [canonical_action(a) for a in prefix_actions],
        "levels_completed_start": levels_start,
        "levels_completed_end": levels_end,
        "levels_completed_start_provenance": levels_start_provenance,
        "levels_completed_end_provenance": levels_end_provenance,
    }


def trajectory_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shape_aware_changed_cells(before: np.ndarray, after: np.ndarray) -> int:
    if before.shape != after.shape:
        # Matches the targeted explorer's documented fallback for shape changes.
        return int(after.size)
    return int(np.count_nonzero(before != after))


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
