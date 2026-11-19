import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction


GAME_ID = "ar25-0c556536"
SEED = 0
MAX_STEPS = 5

# Edit this sequence when running different experiments.
ACTION_SEQUENCE = [
    {"action": "ACTION1", "data": {}},
    {"action": "ACTION2", "data": {}},
    {"action": "ACTION3", "data": {}},
    {"action": "ACTION4", "data": {}},
    {"action": "ACTION5", "data": {}},
]

OUTPUT = Path("data/raw/trajectories")
FRAME_OUTPUT = OUTPUT / "sequence_frames"

TERMINAL_STATES = {"WIN", "GAME_OVER"}


def save_frames(frames, trajectory_id, step, label):
    filenames = []

    for index, frame in enumerate(frames):
        filename = (
            f"{trajectory_id}_step{step:03d}_{label}_{index:02d}.npy"
        )
        np.save(FRAME_OUTPUT / filename, frame)
        filenames.append(filename)

    return filenames


def state_name(observation):
    return observation.state.value


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FRAME_OUTPUT.mkdir(parents=True, exist_ok=True)

    trajectory_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    arc = Arcade()
    env = arc.make(
        game_id=GAME_ID,
        seed=SEED,
        save_recording=False,
        include_frame_data=True,
    )

    if env is None:
        raise RuntimeError("Could not load AR25")

    current = env.reset()

    if current is None:
        raise RuntimeError("Environment reset failed")

    initial_files = save_frames(
        current.frame, trajectory_id, 0, "initial"
    )

    records = []

    for step_number, action_spec in enumerate(
        ACTION_SEQUENCE[:MAX_STEPS], start=1
    ):
        current_state = state_name(current)

        if current_state in TERMINAL_STATES:
            print(f"Terminal state reached: {current_state}")
            break

        action_name = action_spec["action"]
        action_data = action_spec.get("data", {})

        if action_name not in GameAction.__members__:
            raise ValueError(f"Unknown action: {action_name}")

        if action_name == "RESET":
            raise ValueError(
                "RESET is not allowed inside a sequential trajectory"
            )

        before_files = save_frames(
            current.frame, trajectory_id, step_number, "before"
        )

        before_frames = current.frame

        result = env.step(
            GameAction[action_name],
            data=action_data,
        )

        if result is None:
            raise RuntimeError(
                f"{action_name} returned no observation at step "
                f"{step_number}"
            )

        after_files = save_frames(
            result.frame, trajectory_id, step_number, "after"
        )

        changed_pixels = []

        for before_frame, after_frame in zip(
            before_frames, result.frame
        ):
            if before_frame.shape != after_frame.shape:
                changed_pixels.append(None)
            else:
                changed_pixels.append(
                    int(np.count_nonzero(before_frame != after_frame))
                )

        record = {
            "trajectory_id": trajectory_id,
            "step": step_number,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "game_id": GAME_ID,
            "seed": SEED,
            "action": action_name,
            "action_data": action_data,
            "before_files": before_files,
            "after_files": after_files,
            "before_frame_count": len(before_frames),
            "after_frame_count": len(result.frame),
            "changed_pixels_per_frame": changed_pixels,
            "before_state": current_state,
            "after_state": state_name(result),
            "levels_completed": result.levels_completed,
            "win_levels": result.win_levels,
            "available_actions": result.available_actions,
            "full_reset": result.full_reset,
            "action_input": (
                result.action_input.model_dump(mode="json")
                if result.action_input is not None
                else None
            ),
        }

        records.append(record)

        print(
            f"Step {step_number}: {action_name} | "
            f"{record['before_state']} -> {record['after_state']} | "
            f"Changed pixels: {changed_pixels}"
        )

        current = result

        if state_name(current) in TERMINAL_STATES:
            print(f"Stopping at terminal state: {state_name(current)}")
            break

    trajectory_record = {
        "trajectory_id": trajectory_id,
        "game_id": GAME_ID,
        "seed": SEED,
        "initial_files": initial_files,
        "step_count": len(records),
        "final_state": state_name(current),
        "steps": records,
    }

    jsonl_path = OUTPUT / "sequences.jsonl"

    with jsonl_path.open("a") as file:
        file.write(json.dumps(trajectory_record) + "\n")

    print("\nTRAJECTORY COMPLETE")
    print(json.dumps(trajectory_record, indent=2))
    print(f"\nSaved to: {jsonl_path}")


if __name__ == "__main__":
    main()
