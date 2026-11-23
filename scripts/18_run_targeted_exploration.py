import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = "ar25-0c556536"
BASE_OUT = Path("data/raw/exploration")

PREFIX_L0_TO_L1 = [
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
    {"action": "ACTION2", "action_data": {}},
]

PREFIX_L1_TO_L2 = [
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION3", "action_data": {}},
    {"action": "ACTION5", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
    {"action": "ACTION2", "action_data": {}},
]


def to_grid(raw_frame) -> np.ndarray:
    arr = np.asarray(raw_frame, dtype=np.int8)
    while arr.ndim > 2:
        arr = arr[-1]
    return np.ascontiguousarray(arr, dtype=np.int8).copy()


def color_transitions(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    if before.shape != after.shape:
        return {}
    diff = before != after
    if not np.any(diff):
        return {}
    b_vals = before[diff].astype(int).tolist()
    a_vals = after[diff].astype(int).tolist()
    out: dict[str, int] = {}
    for bv, av in zip(b_vals, a_vals):
        key = f"{bv}->{av}"
        out[key] = out.get(key, 0) + 1
    return out


def bbox_xyxy(before: np.ndarray, after: np.ndarray):
    if before.shape != after.shape:
        return None
    ys, xs = np.where(before != after)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def estimate_counter_cells(before: np.ndarray, after: np.ndarray) -> int:
    if before.shape != after.shape or before.ndim != 2:
        return 0
    # Top HUD/counter strip above main play area (rows 0..11)
    top_h = min(12, before.shape[0])
    return int(np.count_nonzero(before[:top_h, :] != after[:top_h, :]))


def classify_outcome(
    init_arr: np.ndarray,
    final_arr: np.ndarray,
    levels_start: int,
    levels_end: int,
    any_change: bool,
) -> str:
    if levels_end > levels_start:
        return "level_progress"
    if not any_change:
        return "no_visible_change"
    if init_arr.shape == final_arr.shape and np.array_equal(init_arr, final_arr):
        return "changed_then_returned_to_start"
    return "changed"


def build_targeted_specs(stage: str, seed: int) -> list[dict]:
    specs: list[dict] = []
    # Active/sampled coordinates across playable & UI regions
    focus_coords = [
        (45, 18), (24, 24), (32, 32), (18, 28), (40, 28),
        (20, 40), (36, 42), (48, 36), (14, 18), (52, 24),
    ]
    dirs = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    if stage == "level0_core":
        # 1. Level progression sequences (0 -> 1)
        specs.append({
            "experiment_id": "L0_PROG_01",
            "category": "level_progression",
            "question": f"On seed {seed} from reset, does executing 10x ACTION2 followed by 5x ACTION3 advance levels_completed?",
            "prefix": [],
            "actions": PREFIX_L0_TO_L1,
        })
        specs.append({
            "experiment_id": "L0_PROG_02",
            "category": "level_progression",
            "question": f"On seed {seed} from reset, how does executing 5x ACTION3 before 10x ACTION2 compare in changed cells and level progress?",
            "prefix": [],
            "actions": [
                {"action": "ACTION3", "action_data": {}} for _ in range(5)
            ] + [{"action": "ACTION2", "action_data": {}} for _ in range(10)],
        })

        # 2. Reversible / undo sequences with ACTION7
        for i, d in enumerate(dirs):
            specs.append({
                "experiment_id": f"L0_UNDO_{i:02d}",
                "category": "reversible_undo",
                "question": f"On seed {seed} from reset, what happens after two {d} steps followed by ACTION7 undo?",
                "prefix": [],
                "actions": [
                    {"action": d, "action_data": {}},
                    {"action": d, "action_data": {}},
                    {"action": "ACTION7", "action_data": {}},
                ],
            })

        # 3. Selection cycling (ACTION5) + directional move
        for i, d in enumerate(dirs):
            specs.append({
                "experiment_id": f"L0_SEL_{i:02d}",
                "category": "selection_cycle",
                "question": f"On seed {seed} from reset, what changes when ACTION5 is invoked before {d}?",
                "prefix": [],
                "actions": [
                    {"action": "ACTION5", "action_data": {}},
                    {"action": d, "action_data": {}},
                ],
            })

        # 4. Targeted ACTION6 clicks + step response
        for i, (x, y) in enumerate(focus_coords):
            d = dirs[i % len(dirs)]
            specs.append({
                "experiment_id": f"L0_CLK_{i:02d}",
                "category": "targeted_click",
                "question": f"On seed {seed} from reset, does clicking ({x},{y}) with ACTION6 and then running {d} modify pixels?",
                "prefix": [],
                "actions": [
                    {"action": "ACTION6", "action_data": {"x": x, "y": y}},
                    {"action": d, "action_data": {}},
                ],
            })

        # 5. Boundary / repeated single-direction pushes
        for i, d in enumerate(dirs):
            specs.append({
                "experiment_id": f"L0_PUSH_{i:02d}",
                "category": "boundary_push",
                "question": f"On seed {seed} from reset, what is the cumulative net pixel delta of 6 consecutive {d} steps?",
                "prefix": [],
                "actions": [{"action": d, "action_data": {}} for _ in range(6)],
            })

    elif stage == "level1_branch":
        # Branch from verified Level 1 starting state
        for i, d in enumerate(dirs):
            specs.append({
                "experiment_id": f"L1_MOVE_{i:02d}",
                "category": "level1_movement",
                "question": f"Starting in Level 1 on seed {seed}, what does a 3-step {d} sequence change?",
                "prefix": PREFIX_L0_TO_L1,
                "actions": [{"action": d, "action_data": {}} for _ in range(3)],
            })

        for i, (x, y) in enumerate(focus_coords):
            specs.append({
                "experiment_id": f"L1_CLK_{i:02d}",
                "category": "level1_click",
                "question": f"Starting in Level 1 on seed {seed}, what color transitions occur after ACTION6 at ({x},{y}) and ACTION5?",
                "prefix": PREFIX_L0_TO_L1,
                "actions": [
                    {"action": "ACTION6", "action_data": {"x": x, "y": y}},
                    {"action": "ACTION5", "action_data": {}},
                ],
            })

        # Level 1 -> Level 2 progression variants
        specs.append({
            "experiment_id": "L1_TO_L2_ACT5",
            "category": "level_progression",
            "question": f"Starting in Level 1 on seed {seed}, does 2x ACTION3, ACTION5, and 8x ACTION2 advance to Level 2?",
            "prefix": PREFIX_L0_TO_L1,
            "actions": [
                {"action": "ACTION3", "action_data": {}},
                {"action": "ACTION3", "action_data": {}},
                {"action": "ACTION5", "action_data": {}},
            ] + [{"action": "ACTION2", "action_data": {}} for _ in range(8)],
        })
        specs.append({
            "experiment_id": "L1_TO_L2_CLK",
            "category": "level_progression",
            "question": f"Starting in Level 1 on seed {seed}, does 2x ACTION3, ACTION6 at (45,18), and 8x ACTION2 advance to Level 2?",
            "prefix": PREFIX_L0_TO_L1,
            "actions": [
                {"action": "ACTION3", "action_data": {}},
                {"action": "ACTION3", "action_data": {}},
                {"action": "ACTION6", "action_data": {"x": 45, "y": 18}},
            ] + [{"action": "ACTION2", "action_data": {}} for _ in range(8)],
        })

    elif stage == "level2_branch":
        prefix_to_l2 = PREFIX_L0_TO_L1 + PREFIX_L1_TO_L2
        for i, d in enumerate(dirs):
            specs.append({
                "experiment_id": f"L2_DIR_{i:02d}",
                "category": "level2_movement",
                "question": f"Starting in Level 2 on seed {seed}, what is the net pixel and bounding-box change after 4x {d}?",
                "prefix": prefix_to_l2,
                "actions": [{"action": d, "action_data": {}} for _ in range(4)],
            })
            specs.append({
                "experiment_id": f"L2_UNDO_{i:02d}",
                "category": "level2_undo",
                "question": f"Starting in Level 2 on seed {seed}, does {d} followed by ACTION7 restore the Level 2 starting frame?",
                "prefix": prefix_to_l2,
                "actions": [
                    {"action": d, "action_data": {}},
                    {"action": "ACTION7", "action_data": {}},
                ],
            })

        for i, (x, y) in enumerate(focus_coords):
            specs.append({
                "experiment_id": f"L2_CLK_{i:02d}",
                "category": "level2_click",
                "question": f"Starting in Level 2 on seed {seed}, what happens after ACTION6 at ({x},{y}) followed by ACTION2?",
                "prefix": prefix_to_l2,
                "actions": [
                    {"action": "ACTION6", "action_data": {"x": x, "y": y}},
                    {"action": "ACTION2", "action_data": {}},
                ],
            })

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["level0_core", "level1_branch", "level2_branch"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{args.stage}_s{args.seed}"
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

    specs = build_targeted_specs(args.stage, args.seed)
    exp_lines: list[str] = []

    for spec in specs:
        obs0 = env.reset()
        if obs0 is None or not obs0.frame:
            raise RuntimeError("Reset returned no observation/frame")

        cur_arr = to_grid(obs0.frame[-1])
        cur_state = obs0.state.value if hasattr(obs0.state, "value") else str(obs0.state)
        raw_levels_reset = getattr(obs0, "levels_completed", None)
        if raw_levels_reset is None:
            raise RuntimeError("Reset observation is missing levels_completed; refusing to assume level 0")
        cur_levels = int(raw_levels_reset)
        levels_completed_reset = cur_levels
        prefix_actions = [dict(a) for a in (spec.get("prefix") or [])]

        # Advance through any starting-state prefix without recording it as target actions
        for p_act in spec.get("prefix") or []:
            p_obs = env.step(GameAction[p_act["action"]], data=p_act.get("action_data") or {})
            if p_obs is None or not p_obs.frame:
                raise RuntimeError("Prefix step returned no frame")
            cur_arr = to_grid(p_obs.frame[-1])
            cur_state = p_obs.state.value if hasattr(p_obs.state, "value") else str(p_obs.state)
            raw_prefix_levels = getattr(p_obs, "levels_completed", None)
            if raw_prefix_levels is not None:
                cur_levels = int(raw_prefix_levels)

        eid = spec["experiment_id"]
        init_arr = cur_arr.copy()
        init_fname = f"{eid}_step00_start.npy"
        np.save(frames_dir / init_fname, init_arr)

        levels_start = cur_levels
        steps_meta = []
        any_change = False
        total_changed = 0
        total_counter = 0

        for idx, act_spec in enumerate(spec["actions"], start=1):
            before_fname = f"{eid}_step{idx:02d}_before.npy"
            np.save(frames_dir / before_fname, cur_arr)

            act_name = act_spec["action"]
            act_data = act_spec.get("action_data") or {}
            obs_after = env.step(GameAction[act_name], data=act_data)
            if obs_after is None or not obs_after.frame:
                raise RuntimeError(f"Step {idx} ({act_name}) returned no frame")

            after_files = []
            for f_i, raw_f in enumerate(obs_after.frame):
                arr_f = to_grid(raw_f)
                fname = f"{eid}_step{idx:02d}_after_{f_i:02d}.npy"
                np.save(frames_dir / fname, arr_f)
                after_files.append(fname)

            final_step_arr = to_grid(obs_after.frame[-1]).copy()
            changed = int(np.count_nonzero(cur_arr != final_step_arr)) if cur_arr.shape == final_step_arr.shape else int(final_step_arr.size)
            c_cells = estimate_counter_cells(cur_arr, final_step_arr)
            if changed > 0:
                any_change = True
            total_changed += changed
            total_counter += c_cells

            state_after = obs_after.state.value if hasattr(obs_after.state, "value") else str(obs_after.state)
            levels_before = cur_levels
            raw_levels_after = getattr(obs_after, "levels_completed", None)
            levels_after = cur_levels if raw_levels_after is None else int(raw_levels_after)
            avail = [a.name if hasattr(a, "name") else str(a) for a in (getattr(obs_after, "available_actions", None) or [])]

            steps_meta.append({
                "step": idx,
                "action": act_name,
                "action_data": act_data,
                "before_file": before_fname,
                "after_files": after_files,
                "after_frame_count": len(after_files),
                "state_before": cur_state,
                "state_after": state_after,
                "levels_completed": levels_after,  # backward-compatible alias for post-step value
                "levels_completed_before": levels_before,
                "levels_completed_after": levels_after,
                "counter_cells_used": c_cells,
                "play_area_changed_cells": max(0, changed - c_cells),
                "available_actions": avail,
                "full_reset": False,
                "shape_changed": cur_arr.shape != final_step_arr.shape,
                "changed_cells": changed,
                "bbox_xyxy": bbox_xyxy(cur_arr, final_step_arr),
                "color_transitions": color_transitions(cur_arr, final_step_arr),
            })
            cur_arr = final_step_arr.copy()
            cur_state = state_after
            cur_levels = levels_after

        outcome = classify_outcome(init_arr, cur_arr, levels_start, cur_levels, any_change)
        rec = {
            "run_id": run_id,
            "experiment_id": eid,
            "category": spec["category"],
            "question": spec["question"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "game_id": GAME_ID,
            "seed": args.seed,
            "actions": spec["actions"],  # target actions; prefix is kept separate
            "prefix_actions": prefix_actions,
            "prefix_actions_provenance": "observed",
            "levels_completed_reset": levels_completed_reset,
            "levels_completed_after_prefix": levels_start,
            "levels_completed_start": levels_start,
            "levels_completed_start_provenance": "observed",
            "levels_completed_end": cur_levels,
            "levels_completed_end_provenance": "observed",
            "steps_executed": len(steps_meta),
            "steps": steps_meta,
            "net_change": {
                "shape_changed": init_arr.shape != cur_arr.shape,
                "changed_cells": int(np.count_nonzero(init_arr != cur_arr)) if init_arr.shape == cur_arr.shape else int(cur_arr.size),
                "bbox_xyxy": bbox_xyxy(init_arr, cur_arr),
                "color_transitions": color_transitions(init_arr, cur_arr),
            },
            "net_change_play_area": {
                "shape_changed": init_arr.shape != cur_arr.shape,
                "changed_cells": total_changed,
                "bbox_xyxy": bbox_xyxy(init_arr, cur_arr),
                "color_transitions": color_transitions(init_arr, cur_arr),
            },
            "counter_cells_used": total_counter,
            "final_state": cur_state,
            "outcome": outcome,
            "error": None,
        }
        exp_lines.append(json.dumps(rec))

    (run_dir / "experiments.jsonl").write_text("\n".join(exp_lines) + "\n")
    print(f"RUN_DIR={run_dir} STAGE={args.stage} RECORDED={len(exp_lines)}")


if __name__ == "__main__":
    main()
