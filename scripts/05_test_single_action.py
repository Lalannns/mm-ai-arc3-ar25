from pathlib import Path
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = "ar25-0c556536"

arc = Arcade()
env = arc.make(
    game_id=GAME_ID,
    seed=0,
    save_recording=False,
    include_frame_data=True,
)

if env is None:
    raise RuntimeError("Could not load AR25")

before = env.reset()
if before is None:
    raise RuntimeError("Reset failed")

output = Path("data/raw/experiments")
output.mkdir(parents=True, exist_ok=True)

np.save(output / "action1_before.npy", before.frame[0])

after = env.step(GameAction.ACTION1, data={})
if after is None:
    raise RuntimeError("ACTION1 returned no observation")

np.save(output / "action1_after.npy", after.frame[0])

print("Action:", GameAction.ACTION1)
print("Before state:", before.state)
print("After state:", after.state)
print("Levels completed:", after.levels_completed)
print("Available actions:", after.available_actions)
print("Changed pixels:", np.count_nonzero(
    before.frame[0] != after.frame[0]
))
print("Saved before/after observations.")
