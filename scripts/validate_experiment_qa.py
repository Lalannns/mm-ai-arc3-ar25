import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PARQUET_PATH = Path("data/parquet/ar25_experiment_qa.parquet")
SCHEMA_PATH = Path("schemas/experiment_qa_schema.json")
TARGET_PAIRS = 1000


def main() -> None:
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(f"Missing Parquet file: {PARQUET_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text())
    required_keys = set(schema["required"])
    allowed_keys = set(schema["properties"].keys())

    rows = pq.read_table(PARQUET_PATH).to_pylist()
    errors: list[str] = []
    seen_sigs: set[str] = set()
    valid_unique_count = 0

    for idx, row in enumerate(rows):
        row_keys = set(row.keys())
        missing = required_keys - row_keys
        extra = row_keys - allowed_keys
        if missing:
            errors.append(f"Row {idx} missing required keys: {sorted(missing)}")
            continue
        if extra:
            errors.append(f"Row {idx} has unexpected keys: {sorted(extra)}")
            continue

        sig = row.get("signature_hash", "")
        if not isinstance(sig, str) or len(sig) != 64:
            errors.append(f"Row {idx} invalid signature_hash")
            continue
        if sig in seen_sigs:
            errors.append(f"Row {idx} duplicate signature_hash: {sig}")
            continue
        seen_sigs.add(sig)

        if not row.get("question") or not row.get("answer"):
            errors.append(f"Row {idx} missing non-empty question or answer")
            continue
        if row.get("answer_grounding") != "observed_sequence_metadata":
            errors.append(f"Row {idx} invalid answer_grounding: {row.get('answer_grounding')}")
            continue
        if not isinstance(row.get("step_count"), int) or row["step_count"] < 1:
            errors.append(f"Row {idx} invalid step_count: {row.get('step_count')}")
            continue

        try:
            actions = json.loads(row["action_sequence"])
            inter_paths = json.loads(row["intermediate_frame_paths"])
        except Exception as exc:
            errors.append(f"Row {idx} JSON field parse error: {exc}")
            continue

        if not isinstance(actions, list) or len(actions) != row["step_count"]:
            errors.append(f"Row {idx} action_sequence length mismatch")
            continue

        frame_paths = [row["initial_frame_path"], *inter_paths, row["final_frame_path"]]
        frames_ok = True
        for fp_str in frame_paths:
            fp = Path(fp_str)
            if not fp.is_file():
                errors.append(f"Row {idx} missing frame file: {fp}")
                frames_ok = False
                break
            try:
                arr = np.load(fp)
                if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
                    errors.append(f"Row {idx} invalid frame dimensions in {fp}: {arr.shape}")
                    frames_ok = False
                    break
            except Exception as exc:
                errors.append(f"Row {idx} failed decoding frame {fp}: {exc}")
                frames_ok = False
                break

        if frames_ok:
            valid_unique_count += 1

    print(f"Total rows read: {len(rows)}")
    print(f"Conservative verified unique QA pairs: {valid_unique_count}")
    print(f"Target requirement: {TARGET_PAIRS} (shortfall: {max(0, TARGET_PAIRS - valid_unique_count)})")
    print(f"Validation errors: {len(errors)}")
    for err in errors[:20]:
        print("  ERROR:", err)


if __name__ == "__main__":
    main()
