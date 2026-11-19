import json
from pathlib import Path
from collections import Counter
import numpy as np

folder = Path("data/raw/trajectories")

records = [
    json.loads(line)
    for line in (folder / "transitions.jsonl").read_text().splitlines()
]

records = [
    r for r in records
    if any(r["experiment_id"].endswith("_" + a)
           for a in ["ACTION1", "ACTION2", "ACTION3",
                     "ACTION4", "ACTION5", "ACTION7"])
]

for r in records[-6:]:
    before = np.load(folder / r["before_files"][0])
    after = np.load(folder / r["after_files"][0])

    mask = before != after
    ys, xs = np.where(mask)

    print("\n", r["action"])
    print("Changed pixels:", len(xs))

    if len(xs):
        print("Bounding box:",
              (int(xs.min()), int(ys.min()),
               int(xs.max()), int(ys.max())))

        transitions = Counter(
            zip(before[mask].tolist(), after[mask].tolist())
        )
        print("Transitions:", dict(transitions))
    else:
        print("No pixel changes.")
