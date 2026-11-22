import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path("data/raw/exploration")
OUT_PATH = Path("data/parquet/ar25_experiment_qa.parquet")


def iter_run_folders(base_dir: Path) -> list[str]:
    if not base_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in base_dir.iterdir()
        if p.is_dir() and (p / "experiments.jsonl").is_file()
    )


def format_action(step: dict) -> str:
    act = step["action"]
    data = step.get("action_data") or {}
    if data:
        return f"{act}({json.dumps(data, sort_keys=True)})"
    return act


def build_factual_answer(exp: dict, actions_fmt: list[str], total_changed: int, levels_start, levels_end) -> str:
    steps = exp["steps"]
    outcome = exp.get("outcome", "unknown")
    final_state = exp.get("final_state", "unknown")
    seq_str = " -> ".join(actions_fmt)
    if len(steps) == 1:
        s0 = steps[0]
        step_changed = int(s0.get("changed_cells", 0))
        return (
            f"Executed {seq_str} (1 step). Recorded outcome: {outcome}; "
            f"state changed from {s0.get('state_before')} to {s0.get('state_after')} "
            f"(final_state={final_state}); changed_cells={step_changed}; "
            f"levels_completed={levels_start}->{levels_end}."
        )
    return (
        f"Executed {seq_str} ({len(steps)} steps). Recorded outcome: {outcome}; "
        f"final_state={final_state}; cumulative changed_cells={total_changed}; "
        f"levels_completed={levels_start}->{levels_end}."
    )


def main() -> None:
    folders = iter_run_folders(BASE_DIR)
    seen_signatures: set[str] = set()
    rows: list[dict] = []

    for folder in folders:
        run_dir = BASE_DIR / folder
        exp_file = run_dir / "experiments.jsonl"
        for idx, line in enumerate(exp_file.read_text().splitlines()):
            if not line.strip():
                continue
            exp = json.loads(line)
            eid = exp["experiment_id"]
            steps = exp.get("steps") or []
            if not steps:
                raise ValueError(f"Experiment {eid} in {folder} has no steps")

            init_rel = f"frames/{eid}_step00_start.npy"
            init_path = run_dir / init_rel
            if not init_path.is_file():
                raise FileNotFoundError(f"Missing initial frame: {init_path}")
            init_arr = np.load(init_path)
            ih = hashlib.sha256(init_path.read_bytes()).hexdigest()

            actions_sig = tuple(
                (s["action"], json.dumps(s.get("action_data") or {}, sort_keys=True))
                for s in steps
            )
            final_rel = f"frames/{steps[-1]['after_files'][-1]}"
            final_path = run_dir / final_rel
            if not final_path.is_file():
                raise FileNotFoundError(f"Missing final frame: {final_path}")
            final_arr = np.load(final_path)
            fh = hashlib.sha256(final_path.read_bytes()).hexdigest()

            sig_payload = json.dumps(
                {
                    "question": exp["question"],
                    "initial_sha256": ih,
                    "actions": [list(a) for a in actions_sig],
                    "final_sha256": fh,
                    "outcome": exp.get("outcome"),
                    "final_state": exp.get("final_state"),
                },
                sort_keys=True,
            )
            sig_hash = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()
            if sig_hash in seen_signatures:
                continue
            seen_signatures.add(sig_hash)

            intermediate_rels: list[str] = []
            total_changed = 0
            for s_i, s in enumerate(steps):
                total_changed += int(s.get("changed_cells", 0))
                after_files = s.get("after_files") or []
                for af in after_files:
                    rel = f"frames/{af}"
                    fp = run_dir / rel
                    if not fp.is_file():
                        raise FileNotFoundError(f"Missing frame {fp} in {eid}")
                    arr = np.load(fp)
                    if arr.ndim != 2:
                        raise ValueError(f"Unexpected frame ndim {arr.ndim} in {fp}")
                    if s_i < len(steps) - 1 or af != steps[-1]["after_files"][-1]:
                        intermediate_rels.append(str(fp))

            levels_start = steps[0].get("state_before") and steps[0].get("levels_completed")
            levels_end = steps[-1].get("levels_completed")
            actions_fmt = [format_action(s) for s in steps]
            answer = build_factual_answer(exp, actions_fmt, total_changed, levels_start, levels_end)

            rows.append(
                {
                    "record_id": f"{folder}_{eid}_{idx:03d}",
                    "source_run": folder,
                    "experiment_id": eid,
                    "signature_hash": sig_hash,
                    "question": exp["question"],
                    "answer": answer,
                    "answer_grounding": "observed_sequence_metadata",
                    "step_count": len(steps),
                    "action_sequence": json.dumps(actions_fmt),
                    "outcome": str(exp.get("outcome", "")),
                    "final_state": exp.get("final_state"),
                    "levels_completed_start": levels_start if levels_start is None else int(levels_start),
                    "levels_completed_end": levels_end if levels_end is None else int(levels_end),
                    "total_changed_cells": int(total_changed),
                    "initial_frame_path": str(init_path),
                    "intermediate_frame_paths": json.dumps(intermediate_rels),
                    "final_frame_path": str(final_path),
                    "frame_shape": f"{init_arr.shape[0]}x{init_arr.shape[1]}",
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT_PATH)
    print(f"Scanned {len(folders)} runs; wrote {len(rows)} unique experiment QA rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
