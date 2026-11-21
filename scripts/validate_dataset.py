
import argparse
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from PIL import Image

RAW = Path("data/raw")
TRAJ = RAW / "trajectories"
FRAMES = TRAJ / "sequence_frames"
EXPLORATION = RAW / "exploration"
SCHEMA_PATH = Path("schemas/trajectory_schema.json")
REPORT_DIR = Path("reports")
GRID_SHAPE = (64, 64)
IMAGE_SIZE = (512, 512)


class Findings:
    def __init__(self):
        self.items = []
        self.stats = {}

    def add(self, level, check, target, message):
        self.items.append(
            {"level": level, "check": check, "target": str(target), "message": message}
        )

    def error(self, check, target, message):
        self.add("ERROR", check, target, message)

    def warn(self, check, target, message):
        self.add("WARNING", check, target, message)

    def count(self, level):
        return sum(1 for f in self.items if f["level"] == level)


def read_jsonl(path, findings):
    records = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            findings.error("malformed_json", f"{path}:{number}", str(exc))
    return records


def check_frame(path, findings, cache):
    if path in cache:
        return cache[path]

    if not path.exists():
        findings.error("missing_file", path, "referenced frame file does not exist")
        cache[path] = None
        return None

    try:
        frame = np.load(path)
    except Exception as exc:
        findings.error("unreadable_file", path, str(exc))
        cache[path] = None
        return None

    if frame.shape != GRID_SHAPE:
        findings.error("dimensions", path, f"shape {frame.shape}, expected {GRID_SHAPE}")
    elif frame.min() < 0 or frame.max() > 15:
        findings.error(
            "value_range", path,
            f"values {frame.min()}..{frame.max()}, expected 0..15"
        )

    cache[path] = frame
    return frame


def check_changed_pixels(before, after, recorded, target, findings):
    if before is None or after is None or recorded is None:
        return

    if before.shape != after.shape:
        return

    actual = int(np.count_nonzero(before != after))

    if actual != recorded:
        findings.error(
            "changed_pixels_mismatch",
            target,
            f"log says {recorded}, frames give {actual}",
        )


def check_sequences(findings, referenced, cache):
    path = TRAJ / "sequences.jsonl"

    if not path.exists():
        findings.warn(
            "missing_file", path,
            "no sequential trajectories recorded yet"
        )
        return 0, 0

    trajectories = read_jsonl(path, findings)
    seen_ids = Counter(t.get("trajectory_id") for t in trajectories)

    for tid, n in seen_ids.items():
        if n > 1:
            findings.error(
                "duplicate_record", path,
                f"trajectory_id {tid} appears {n} times"
            )

    steps_total = 0

    for traj in trajectories:
        tid = traj.get("trajectory_id", "?")

        for key in ("trajectory_id", "game_id", "steps", "initial_files"):
            if key not in traj:
                findings.error(
                    "missing_field", f"{path}:{tid}",
                    f"field '{key}' missing"
                )

        for name in traj.get("initial_files", []):
            referenced.add(FRAMES / name)
            check_frame(FRAMES / name, findings, cache)

        steps = traj.get("steps", [])
        steps_total += len(steps)

        if traj.get("step_count") != len(steps):
            findings.error(
                "incomplete_transition", f"{path}:{tid}",
                f"step_count={traj.get('step_count')} but {len(steps)} steps stored",
            )

        numbers = [s.get("step") for s in steps]

        if numbers != list(range(1, len(steps) + 1)):
            findings.error(
                "step_order", f"{path}:{tid}",
                f"steps are {numbers}, expected 1..{len(steps)}"
            )

        previous_after = None

        for s in steps:
            target = f"{path}:{tid}:step{s.get('step')}"
            before_files = s.get("before_files") or []
            after_files = s.get("after_files") or []

            if not before_files or not after_files:
                findings.error(
                    "incomplete_transition", target,
                    "before or after frame list is empty"
                )
                continue

            for name in before_files + after_files:
                referenced.add(FRAMES / name)

            befores = [
                check_frame(FRAMES / n, findings, cache)
                for n in before_files
            ]
            afters = [
                check_frame(FRAMES / n, findings, cache)
                for n in after_files
            ]

            for b, a, recorded in zip(
                befores, afters, s.get("changed_pixels_per_frame") or []
            ):
                check_changed_pixels(b, a, recorded, target, findings)

            if previous_after is not None and befores[-1] is not None:
                if not np.array_equal(previous_after, befores[-1]):
                    findings.error(
                        "broken_sequence", target,
                        "before frame differs from previous step's after frame",
                    )

            previous_after = afters[-1]

    return len(trajectories), steps_total


def check_transitions(findings, referenced, cache):
    path = TRAJ / "transitions.jsonl"

    if not path.exists():
        return 0

    records = read_jsonl(path, findings)
    ids = Counter(r.get("experiment_id") for r in records)

    for eid, n in ids.items():
        if n > 1:
            findings.error(
                "duplicate_record", path,
                f"experiment_id {eid} appears {n} times"
            )

    for r in records:
        target = f"{path}:{r.get('experiment_id')}"

        before = r.get("before_files") or (
            [r["before_file"]] if "before_file" in r else []
        )
        after = r.get("after_files") or (
            [r["after_file"]] if "after_file" in r else []
        )

        if "before_file" in r:
            findings.warn(
                "legacy_format", target,
                "uses before_file/after_file instead of *_files lists"
            )

        if not before or not after:
            findings.error(
                "incomplete_transition", target,
                "before or after frame missing"
            )
            continue

        for name in before + after:
            referenced.add(TRAJ / name)

        b = check_frame(TRAJ / before[-1], findings, cache)
        a = check_frame(TRAJ / after[-1], findings, cache)

        recorded = r.get("changed_pixels")

        if recorded is None and r.get("changed_pixels_per_after_frame"):
            recorded = r["changed_pixels_per_after_frame"][-1]

        check_changed_pixels(b, a, recorded, target, findings)

    return len(records)


def check_episode_logs(findings):
    total = 0

    for path in sorted((RAW / "logs").glob("*.json")):
        try:
            steps = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            findings.error("malformed_json", path, str(exc))
            continue

        total += len(steps)
        numbers = [s.get("step") for s in steps]

        if numbers != list(range(len(steps))):
            findings.error(
                "step_order", path,
                f"steps are not 0..{len(steps) - 1}"
            )

        dupes = [n for n, c in Counter(numbers).items() if c > 1]

        if dupes:
            findings.error(
                "duplicate_record", path,
                f"step numbers repeated: {dupes}"
            )

        for i, s in enumerate(steps):
            target = f"{path}:step{s.get('step')}"
            grid = np.array(s.get("grid_state"))

            if grid.ndim == 3:
                grid = grid[-1]

            if grid.shape != GRID_SHAPE:
                findings.error(
                    "dimensions", target,
                    f"grid_state shape {grid.shape}"
                )

            has_next = s.get("next_grid_state") is not None

            if not has_next and i == len(steps) - 1:
                findings.warn(
                    "incomplete_transition", target,
                    "last step has no next_grid_state, so its after frame is unknown",
                )

            if has_next:
                after = np.array(s["next_grid_state"])

                if after.shape != GRID_SHAPE:
                    findings.error(
                        "dimensions", target,
                        f"next_grid_state shape {after.shape}"
                    )

            image = s.get("image_path")

            if image and not Path(image).exists():
                findings.error(
                    "missing_file", image,
                    f"referenced by {target}"
                )

            recorded = (s.get("diff") or {}).get("total_changed_pixels")

            if has_next and recorded is not None and grid.shape == GRID_SHAPE:
                check_changed_pixels(
                    grid, np.array(s["next_grid_state"]),
                    recorded, target, findings
                )

    return total


def check_exploration(findings, referenced, cache):
    total_experiments = 0
    total_steps = 0

    if not EXPLORATION.exists():
        findings.warn(
            "missing_file", EXPLORATION,
            "no exploration data found"
        )
        return 0, 0

    for run_folder in sorted(EXPLORATION.iterdir()):
        if not run_folder.is_dir():
            continue

        path = run_folder / "experiments.jsonl"
        frames_folder = run_folder / "frames"

        if not path.exists():
            findings.warn(
                "missing_file", path,
                "no experiments.jsonl in run folder"
            )
            continue

        experiments = read_jsonl(path, findings)
        total_experiments += len(experiments)

        for experiment in experiments:
            eid = experiment.get("experiment_id", "?")
            target = f"{path}:{eid}"
            steps = experiment.get("steps", [])
            total_steps += len(steps)

            numbers = [s.get("step") for s in steps]

            if numbers != list(range(1, len(steps) + 1)):
                findings.error(
                    "step_order", target,
                    f"steps are {numbers}, expected 1..{len(steps)}",
                )

            start_path = frames_folder / f"{eid}_step00_start.npy"
            referenced.add(start_path)
            previous_after = check_frame(start_path, findings, cache)

            for step in steps:
                step_target = f"{target}:step{step.get('step')}"
                after_files = step.get("after_files") or []

                if not after_files:
                    findings.error(
                        "incomplete_transition", step_target,
                        "after_files list is empty",
                    )
                    previous_after = None
                    continue

                checked_frames = []

                for filename in after_files:
                    frame_path = frames_folder / filename
                    referenced.add(frame_path)
                    frame = check_frame(frame_path, findings, cache)
                    checked_frames.append(frame)

                after = checked_frames[-1]

                check_changed_pixels(
                    previous_after,
                    after,
                    step.get("changed_cells"),
                    step_target,
                    findings,
                )

                previous_after = after

    return total_experiments, total_steps


def check_orphans(findings, referenced):
    files = set(TRAJ.rglob("*.npy"))
    orphans = sorted(files - referenced)

    for path in orphans:
        findings.warn(
            "orphan_file", path,
            "frame file is not referenced by any log"
        )

    return len(files), len(orphans)


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value]

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def transition_hash(before, after, action):
    h = hashlib.sha1()
    h.update(before.astype(np.int8).tobytes())
    h.update(str(action).encode())
    h.update(after.astype(np.int8).tobytes())
    return h.hexdigest()[:16]


def check_parquet(path, validator, findings):
    df = pd.read_parquet(path)
    stats = {"rows": len(df), "columns": list(df.columns)}

    for sid, n in df["sample_id"].value_counts().items():
        if n > 1:
            findings.error(
                "duplicate_record", path,
                f"sample_id {sid} appears {n} times"
            )

    seen = set()
    hash_counts = Counter()

    for _, row in df.iterrows():
        record = {k: to_jsonable(v) for k, v in row.items()}
        target = f"{path}:{record.get('sample_id')}"

        for err in sorted(
            validator.iter_errors(record),
            key=lambda e: list(e.path)
        ):
            where = "/".join(str(p) for p in err.path) or "(row)"
            findings.error(
                "schema", target,
                f"{where}: {err.message[:160]}"
            )

        try:
            before = np.array(record["grid_before"])
            after = np.array(record["grid_after"])
        except Exception:
            continue

        if before.shape != GRID_SHAPE or after.shape != GRID_SHAPE:
            continue

        if int(np.count_nonzero(before != after)) != record["changed_pixels"]:
            findings.error(
                "changed_pixels_mismatch", target,
                "changed_pixels does not match the grids"
            )

        th = transition_hash(before, after, record["action"])

        if th != record["transition_hash"]:
            findings.error(
                "hash_mismatch", target,
                "transition_hash does not match the grids"
            )

        hash_counts[th] += 1

        if record["is_duplicate"] != (th in seen):
            findings.error(
                "duplicate_flag", target,
                "is_duplicate flag is inconsistent"
            )

        seen.add(th)

        for column in ("image_before", "image_after"):
            image = record.get(column) or {}

            try:
                img = Image.open(io.BytesIO(image["bytes"]))
                img.verify()

                if img.format != "PNG" or img.size != IMAGE_SIZE:
                    findings.error(
                        "dimensions", f"{target}:{column}",
                        f"{img.format} {img.size}"
                    )

            except Exception as exc:
                findings.error(
                    "unreadable_file", f"{target}:{column}", str(exc)
                )

    duplicates = sum(n - 1 for n in hash_counts.values() if n > 1)

    if duplicates:
        findings.warn(
            "duplicate_transition", path,
            f"{duplicates} rows repeat an identical (before, action, after) transition",
        )

    stats["duplicate_transitions"] = duplicates
    stats["unique_transitions"] = len(hash_counts)
    stats["actions"] = dict(sorted(Counter(df["action"]).items()))
    stats["sources"] = dict(sorted(Counter(df["source"]).items()))
    stats["reasoning_sources"] = dict(Counter(df["reasoning_source"]))
    stats["no_change_rows"] = int((df["changed_pixels"] == 0).sum())

    return stats


def write_report(findings, raw_stats, parquet_stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    errors = findings.count("ERROR")
    warnings = findings.count("WARNING")
    status = "PASS" if errors == 0 else "FAIL"

    (out_dir / "validation_report.json").write_text(json.dumps({
        "generated_utc": now,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "raw": raw_stats,
        "parquet": parquet_stats,
        "findings": findings.items,
    }, indent=2))

    lines = [
        "# AR25 dataset validation report",
        "",
        f"Generated: {now}  ",
        f"Command: `python scripts/validate_dataset.py`  ",
        f"Status: **{status}** ({errors} errors, {warnings} warnings)",
        "",
        "## Raw data (`data/raw/`)",
        "",
        "| Item | Count |",
        "|---|---|",
    ]

    lines += [
        f"| {k.replace('_', ' ')} | {v} |"
        for k, v in raw_stats.items()
    ]

    for path, s in parquet_stats.items():
        lines += [
            "",
            f"## Processed dataset `{path}`",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| rows | {s['rows']} |",
            f"| unique transitions | {s['unique_transitions']} |",
            f"| duplicate transitions | {s['duplicate_transitions']} |",
            f"| rows with no visible change | {s['no_change_rows']} |",
            f"| actions | {', '.join(f'{k}: {v}' for k, v in s['actions'].items())} |",
            f"| sources | {', '.join(f'{k}: {v}' for k, v in s['sources'].items())} |",
            f"| reasoning | {', '.join(f'{k}: {v}' for k, v in s['reasoning_sources'].items())} |",
        ]

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Errors | Warnings |",
        "|---|---|---|",
    ]

    by_check = defaultdict(Counter)

    for f in findings.items:
        by_check[f["check"]][f["level"]] += 1

    all_checks = [
        "missing_file", "unreadable_file", "malformed_json", "missing_field",
        "dimensions", "value_range", "step_order", "broken_sequence",
        "incomplete_transition", "duplicate_record", "duplicate_transition",
        "duplicate_flag", "changed_pixels_mismatch", "hash_mismatch",
        "schema", "orphan_file", "legacy_format",
    ]

    for check in all_checks + sorted(set(by_check) - set(all_checks)):
        c = by_check.get(check, Counter())
        lines.append(
            f"| {check} | {c['ERROR']} | {c['WARNING']} |"
        )

    lines += ["", "## Findings", ""]

    if not findings.items:
        lines.append("No findings.")
    else:
        lines += [
            "| Level | Check | Target | Message |",
            "|---|---|---|---|",
        ]

        for f in findings.items:
            lines.append(
                f"| {f['level']} | {f['check']} | "
                f"`{f['target']}` | {f['message']} |"
            )

    (out_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n"
    )

    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", nargs="*",
        default=sorted(
            str(p) for p in Path("data/processed").glob("*.parquet")
        ),
    )
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    if not RAW.exists():
        sys.exit(
            "Run this script from the repository root (data/raw not found)."
        )

    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)

    validator = Draft202012Validator(
        {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": "#/$defs/transition",
        }
    )

    findings = Findings()
    referenced, cache = set(), {}

    trajectories, sequence_steps = check_sequences(
        findings, referenced, cache
    )
    transitions = check_transitions(findings, referenced, cache)
    episode_steps = check_episode_logs(findings)

    exploration_experiments, exploration_steps = check_exploration(
        findings, referenced, cache
    )

    npy_files, orphans = check_orphans(findings, referenced)

    raw_stats = {
        "sequential_trajectories": trajectories,
        "sequential_steps": sequence_steps,
        "single_action_transitions": transitions,
        "episode_log_steps": episode_steps,
        "exploration_experiments": exploration_experiments,
        "exploration_steps": exploration_steps,
        "npy_frame_files": npy_files,
        "orphan_npy_files": orphans,
    }

    parquet_stats = {}

    for p in args.parquet:
        if not Path(p).exists():
            findings.error("missing_file", p, "parquet file not found")
            continue

        parquet_stats[p] = check_parquet(p, validator, findings)

    status = write_report(
        findings, raw_stats, parquet_stats, Path(args.report_dir)
    )

    print(
        f"Status: {status} "
        f"({findings.count('ERROR')} errors, "
        f"{findings.count('WARNING')} warnings)"
    )

    for f in findings.items:
        print(
            f"  {f['level']:7s} {f['check']:24s} "
            f"{f['target']}  {f['message']}"
        )

    print(
        f"Report written to {args.report_dir}/validation_report.md"
    )

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
