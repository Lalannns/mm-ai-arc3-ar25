from pathlib import Path
from arc_agi import Arcade
import numpy as np

GAME_ID = "ar25-0c556536"

arc = Arcade()
env = arc.make(
    game_id=GAME_ID,
    seed=0,
    save_recording=False,
    include_frame_data=True,
)

obs = env.reset()
frame = obs.frame[0]

print("Shape:", frame.shape)
print("Dtype:", frame.dtype)
print("Min:", frame.min())
print("Max:", frame.max())
print("Unique values:", np.unique(frame))
print("Pixel array:")
print(frame)

np.save("data/raw/ar25_initial.npy", frame)
print("\nSaved original frame without conversion:")
print("data/raw/ar25_initial.npy")
