import numpy as np

frame = np.load("data/raw/ar25_initial.npy")

print("Shape:", frame.shape)
print("Unique values:", np.unique(frame, return_counts=True))

print("\nSampled grid (every 4 pixels):")
for y in range(0, frame.shape[0], 4):
    row = []
    for x in range(0, frame.shape[1], 4):
        row.append(str(int(frame[y, x])).rjust(2))
    print(f"y={y:02d}:", " ".join(row))
