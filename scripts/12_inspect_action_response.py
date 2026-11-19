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

before = env.reset()

after = env.step(
    GameAction.ACTION6,
    data={"x": 36, "y": 15},
)

if after is None:
    raise RuntimeError("No response returned")

print("ACTION INPUT:")
print(after.action_input)

print("\nAVAILABLE ACTIONS:")
print(after.available_actions)

print("\nGAME STATE:")
print(after.state)

print("\nLEVELS COMPLETED:")
print(after.levels_completed)

print("\nFULL RESET:")
print(after.full_reset)

print("\nFRAME COUNT:")
print(len(after.frame))

print("\nFRAME CHANGED:")
print(
    (before.frame[0] != after.frame[0]).sum()
)
