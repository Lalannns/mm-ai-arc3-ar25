import numpy as np
from collections import Counter

frame = np.load("data/raw/ar25_initial.npy")

print("Non-background pixel coordinates by value:\n")

for value in sorted(v for v in np.unique(frame) if v != 9):
    ys, xs = np.where(frame == value)

    print(f"Value {value}: {len(xs)} pixels")
    print(
        "  Bounding box:",
        (int(xs.min()), int(ys.min()),
         int(xs.max()), int(ys.max()))
    )
    print(
        "  Sample coordinates:",
        list(zip(xs[:15].tolist(), ys[:15].tolist()))
    )

print("\nMost common 3x3 regions:")
for y in range(0, 64, 3):
    for x in range(0, 64, 3):
        region = frame[y:y+3, x:x+3]
        count = np.count_nonzero(region != 9)
        if count >= 3:
            print(f"x={x}, y={y}: {count} non-background pixels")
