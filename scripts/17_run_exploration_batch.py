import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = "ar25-0c556536"
BASE_OUT = Path("data/raw/exploration")

LEVEL1_SOLVE_PREFIX = [
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
]


def color_transitions(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    diff = before != after
    if not np.any(diff):
        return {}
    b_vals = before[diff].astype(int)
    a_vals = after[diff].astype(int)
    counts: dict[str, int] = {}
    for bv, av in zip(b_vals.tolist(), a_vals.tolist()):
        key = f"{bv}->{av}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def bbox_xyxy(before: np.ndarray, after: np.ndarray):
    ys, xs = np.where(before != after)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def classify_outcome(init_arr: np.ndarray, final_arr: np.ndarray, levels_start: int, levels_end: int, any_change: bool) -> str:
    if levels_end > levels_start:
        return "level_progress"
    if not any_change:
        return "no_visible_change"
    if np.array_equal(init_arr, final_arr):
        return "changed_then_returned_to_start"
    return "changed"


def build_specs(count: int, seed: int, offset: int) -> list[dict]:
    base_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]
    coords = [
        (x, y)
        for y in range(4, 62, 6)
        for x in range(4, 62, 6)
    ]  # 10x10 = 100 distinct grid coordinates

    all_candidates: list[tuple[str, list[dict]]] = []

    # 1. All 3-step action permutations/tuples (6^3 = 216)
    for seq in itertools.product(base_actions, repeat=3):
        acts = [{"action": a, "action_data": {}} for a in seq]
        q = f"On seed {seed}, what is the observed state transition after executing {' -> '.join(seq)}?"
        all_candidates.append((q, acts))

    # 2. All 4-step directional + modifier sequences (6^4 = 1296)
    for seq in itertools.product(base_actions, repeat=4):
        acts = [{"action": a, "action_data": {}} for a in seq]
        q = f"On seed {seed}, what is the cumulative outcome of the 4-step sequence {' -> '.join(seq)}?"
        all_candidates.append((q, acts))

    # 3. Coordinate click (ACTION6) + follow-up action pairs (100 * 6 = 600)
    for (x, y), follow in itertools.product(coords, base_actions):
        acts = [
            {"action": "ACTION6", "action_data": {"x": x, "y": y}},
            {"action": follow, "action_data": {}},
        ]
        q = f"On seed {seed}, what happens when clicking ({x},{y}) with ACTION6 followed by {follow}?"
        all_candidates.append((q, acts))

    # 4. Pre-move + coordinate click + post-move (6 * 50 * 4 = 1200)
    for pre, (x, y), post in itertools.product(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"], coords[:50], ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]):
        acts = [
            {"action": pre, "action_data": {}},
            {"action": "ACTION6", "action_data": {"x": x, "y": y}},
            {"action": post, "action_data": {}},
        ]
        q = f"On seed {seed}, what is the observed result of {pre}, clicking ({x},{y}), and then {post}?"
        all_candidates.append((q, acts))

    selected = all_candidates[offset : offset + count]
    if len(selected) < count:
        raise ValueError(f"Requested count={count} at offset={offset}, only {len(selected)} available")

    specs = []
    for i, (q, acts) in enumerate(selected):
        specs.append({
            "experiment_id": f"BATCH_S{seed}_O{offset + i:05d}",
            "question": q,
            "actions": acts,
        })
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = BASE_OUT / run_id
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    arc = Arcade()
    env = arc.make(
        game_id=GAME_ID,
        seed=args.seed,
        save_recording=False,
        include_frame_data=True,
    )
    if env is None:
        raise RuntimeError(f"Could not load {GAME_ID}")

    specs = build_specs(args.count, args.seed, args.offset)
    exp_lines: list[str] = []

    for spec in specs:
        obs0 = env.reset()
        if obs0 is None or not obs0.frame:
            raise RuntimeError("Environment reset returned no observation/frame")

        eid = spec["experiment_id"]
        init_arr = np.asarray(obs0.frame[-1], dtype=np.int8)
        init_fname = f"{eid}_step00_start.npy"
        np.save(frames_dir / init_fname, init_arr)

        prev_arr = init_arr
        prev_state = obs0.state.value if hasattr(obs0.state, "value") else str(obs0.state)
        levels_start = int(getattr(obs0, "levels_completed", 0) or 0)

        steps_meta = []
        any_change = False
        total_changed = 0

        for idx, act_spec in enumerate(spec["actions"], start=1):
            act_name = act_spec["action"]
            act_data = act_spec.get("action_data") or {}
            obs_after = env.step(GameAction[act_name], data=act_data)
            if obs_after is None or not obs_after.frame:
                raise RuntimeError(f"Step {idx} ({act_name}) returned no frame")

            after_files = []
            for f_i, raw_f in enumerate(obs_after.frame):
                arr_f = np.asarray(raw_f, dtype=np.int8)
                fname = f"{eid}_step{idx:02d}_after_{f_i:02d}.npy"
                np.save(frames_dir / fname, arr_f)
                after_files.append(fname)

            final_step_arr = np.asarray(obs_after.frame[-1], dtype=np.int8)
            changed = int(np.count_nonzero(prev_arr != final_step_arr))
            if changed > 0:
                any_change = True
            total_changed += changed

            state_after = obs_after.state.value if hasattr(obs_after.state, "value") else str(obs_after.state)
            levels_after = int(getattr(obs_after, "levels_completed", levels_start) or 0)
            avail = [a.name if hasattr(a, "name") else str(a) for a in (getattr(obs_after, "available_actions", None) or [])]

            steps_meta.append({
                "step": idx,
                "action": act_name,
                "action_data": act_data,
                "after_files": after_files,
                "after_frame_count": len(after_files),
                "state_before": prev_state,
                "state_after": state_after,
                "levels_completed": levels_after,
                "counter_cells_used": 0,
                "play_area_changed_cells": changed,
                "available_actions": avail,
                "full_reset": False,
                "shape_changed": prev_arr.shape != final_step_arr.shape,
                "changed_cells": changed,
                "bbox_xyxy": bbox_xyxy(prev_arr, final_step_arr),
                "color_transitions": color_transitions(prev_arr, final_step_arr),
            })
            prev_arr = final_step_arr
            prev_state = state_after

        levels_end = int(steps_meta[-1]["levels_completed"])
        outcome = classify_outcome(init_arr, prev_arr, levels_start, levels_end, any_change)

        rec = {
            "run_id": run_id,
            "experiment_id": eid,
            "question": spec["question"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "game_id": GAME_ID,
            "seed": args.seed,
            "actions": spec["actions"],
            "steps_executed": len(steps_meta),
            "steps": steps_meta,
            "net_change": {
                "shape_changed": init_arr.shape != prev_arr.shape,
                "changed_cells": int(np.count_nonzero(init_arr != prev_arr)),
                "bbox_xyxy": bbox_xyxy(init_arr, prev_arr),
                "color_transitions": color_transitions(init_arr, prev_arr),
            },
            "net_change_play_area": {
                "shape_changed": init_arr.shape != prev_arr.shape,
                "changed_cells": total_changed,
                "bbox_xyxy": bbox_xyxy(init_arr, prev_arr),
                "color_transitions": color_transitions(init_arr, prev_arr),
            },
            "counter_cells_used": 0,
            "final_state": prev_state,
            "outcome": outcome,
            "error": None,
        }
        exp_lines.append(json.dumps(rec))

    (run_dir / "experiments.jsonl").write_text("\n".join(exp_lines) + "\n")
    print(f"RUN_DIR={run_dir}")
    print(f"RECORDED={len(exp_lines)}")


if __name__ == "__main__":
    main()
