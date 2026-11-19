import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = "ar25-0c556536"
SEED = 0

ACTION_NAME = "ACTION6"
ACTION_DATA = {"x": 36, "y": 15}

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

before = env.reset()
if before is None:
    raise RuntimeError("Reset failed")

after = env.step(
    GameAction[ACTION_NAME],
    data=ACTION_DATA,
)

if after is None:
    raise RuntimeError(f"{ACTION_NAME} returned no observation")

experiment_id = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S%fZ"
)


def save_frames(frames, prefix):
    filenames = []
    for i, frame in enumerate(frames):
        filename = f"{experiment_id}_{prefix}_{i:02d}.npy"
        np.save(output / filename, frame)
        filenames.append(filename)
    return filenames


before_files = save_frames(before.frame, "before")
after_files = save_frames(after.frame, "after")

changed_pixels = [
    int(np.count_nonzero(before.frame[0] != frame))
    if before.frame[0].shape == frame.shape else None
    for frame in after.frame
]

record = {
    "experiment_id": experiment_id,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "game_id": GAME_ID,
    "seed": SEED,
    "action": ACTION_NAME,
    "action_data": ACTION_DATA,
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

print(json.dumps(record, indent=2))
print("Transition saved.")
