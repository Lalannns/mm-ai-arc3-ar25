from pathlib import Path
import numpy as np
from PIL import Image

frame = np.load("data/raw/ar25_initial.npy")

# ARC palette: standard colors indexed by ARC color ID.
palette = [
    (0, 0, 0),       # 0 black
    (0, 116, 217),   # 1 blue
    (255, 65, 54),   # 2 red
    (46, 204, 64),   # 3 green
    (255, 220, 0),   # 4 yellow
    (170, 170, 170), # 5 grey
    (240, 18, 190),  # 6 magenta
    (255, 133, 27),  # 7 orange
    (127, 219, 255), # 8 light blue
    (135, 12, 37),   # 9 maroon
    (255, 255, 255), # 10 white
    (0, 0, 0),       # 11 black (placeholder; inspect actual palette)
]

rgb = np.array(palette, dtype=np.uint8)[frame.astype(np.uint8)]

# Enlarge cells for human inspection; preserve original grid.
image = Image.fromarray(rgb, mode="RGB")
image = image.resize((640, 640), Image.Resampling.NEAREST)

output = Path("data/raw/ar25_initial_visual.png")
image.save(output)

print("Saved:", output)
print("Original grid remains unchanged.")
