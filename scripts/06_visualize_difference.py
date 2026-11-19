import numpy as np
from PIL import Image

before = np.load("data/raw/experiments/action1_before.npy")
after = np.load("data/raw/experiments/action1_after.npy")

difference = np.zeros((*before.shape, 3), dtype=np.uint8)
difference[before != after] = [255, 0, 0]

Image.fromarray(difference).resize(
    (640, 640), Image.Resampling.NEAREST
).save("data/raw/experiments/action1_difference.png")

print("Saved action1_difference.png")
