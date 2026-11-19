import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = "ar25-0c556536"
SEED = 0
ACTIONS = [
    "ACTION1", "ACTION2", "ACTION3",
    "ACTION4", "ACTION5", "ACTION7"
]

output = Path("data/raw/trajectories")
output.mkdir(parents=True, exist_ok=True)

arc = Arcade()
env = arc.make(
    game_id=GAME_ID,
    seed=SEED,
    save_recording=False,
    include_frame_data=True,
)

if env is None:
    raise RuntimeError("Could not load AR25")

for action_name in ACTIONS:
    before = env.reset()
    if before is None:
        raise RuntimeError("Reset failed")

    action = GameAction[action_name]
    after = env.step(action, data={})

    if after is None:
        raise RuntimeError(f"{action_name} returned no observation")

    experiment_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_" + action_name
    )

    before_files = []
    after_files = []
    changed_pixels = []

    for i, frame in enumerate(before.frame):
        filename = f"{experiment_id}_before_{i:02d}.npy"
        np.save(output / filename, frame)
        before_files.append(filename)

    for i, frame in enumerate(after.frame):
        filename = f"{experiment_id}_after_{i:02d}.npy"
        np.save(output / filename, frame)
        after_files.append(filename)

        if before.frame[0].shape == frame.shape:
            changed_pixels.append(
                int(np.count_nonzero(before.frame[0] != frame))
            )
        else:
            changed_pixels.append(None)

    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "game_id": GAME_ID,
        "seed": SEED,
        "action": action_name,
        "action_data": {},
        "before_files": before_files,
        "after_files": after_files,
        "before_frame_count": len(before.frame),
        "after_frame_count": len(after.frame),
        "changed_pixels_per_after_frame": changed_pixels,
        "before_state": str(before.state),
        "after_state": str(after.state),
        "levels_completed": after.levels_completed,
        "win_levels": after.win_levels,
        "available_actions": after.available_actions,
        "full_reset": after.full_reset,
    }

    with (output / "transitions.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")

    print(
        f"{action_name}: "
        f"{len(after.frame)} frame(s), "
        f"changed pixels={changed_pixels}, "
        f"state={after.state}"
    )

print("\nBaseline action sweep completed.")
