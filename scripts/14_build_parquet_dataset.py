import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAW = Path("data/raw")
EXPLORATION = RAW / "exploration"
OUT_DIR = Path("data/processed")
DEFAULT_EXPLORATION_RUNS = ["20261120T014608Z", "level2_20261120T020820Z"]

HUD_COL, HUD_ROW = 63, 63
LISA_CACHE = OUT_DIR / "lisa_cache.jsonl"

BACKGROUND = 9
IMAGE_SIZE = 512

PALETTE = np.array(
    [
        (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
        (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
        (127, 219, 255), (135, 12, 37), (255, 255, 255), (60, 60, 60),
        (0, 128, 128), (128, 0, 128), (128, 128, 0), (200, 120, 80),
    ],
    dtype=np.uint8,
)

INSTRUCTION = (
    "You are playing the ARC-AGI-3 game AR25 on a 64x64 grid. "
    "Look at the current frame and decide which action to take next. "
    "Explain your reasoning, then give the action "
    "(for ACTION6, a click, also give the cell as x=<column> y=<row>)."
)


def action_text(action, data):
    if isinstance(data, dict) and "x" in data and "y" in data:
        return f"{action} x={int(data['x'])} y={int(data['y'])}"
    return str(action)


def load_npy(folder, names):
    if not names:
        return None
    path = folder / names[-1]
    if not path.exists():
        print(f"  [warn] missing frame file: {path}")
        return None
    return np.load(path)


def strip_state(value):
    return str(value).replace("GameState.", "") if value is not None else None


def action_id(name):
    try:
        return int(str(name).replace("ACTION", ""))
    except ValueError:
        return None


def from_sequences():
    path = RAW / "trajectories" / "sequences.jsonl"
    frames = RAW / "trajectories" / "sequence_frames"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        traj = json.loads(line)
        for s in traj["steps"]:
            rows.append({
                "source": "sequences.jsonl",
                "episode_id": traj["trajectory_id"],
                "step": s["step"],
                "action": s["action"],
                "action_data": s.get("action_data") or {},
                "grid_before": load_npy(frames, s.get("before_files")),
                "grid_after": load_npy(frames, s.get("after_files")),
                "before_state": strip_state(s.get("before_state")),
                "after_state": strip_state(s.get("after_state")),
                "levels_completed": s.get("levels_completed"),
                "raw_reasoning": (s.get("action_input") or {}).get("reasoning"),
            })
    return rows


def from_transitions():
    path = RAW / "trajectories" / "transitions.jsonl"
    folder = RAW / "trajectories"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        before = t.get("before_files") or ([t["before_file"]] if "before_file" in t else [])
        after = t.get("after_files") or ([t["after_file"]] if "after_file" in t else [])
        rows.append({
            "source": "transitions.jsonl",
            "episode_id": t["experiment_id"],
            "step": 1,
            "action": t["action"],
            "action_data": t.get("action_data") or {},
            "grid_before": load_npy(folder, before),
            "grid_after": load_npy(folder, after),
            "before_state": strip_state(t.get("before_state")),
            "after_state": strip_state(t.get("after_state")),
            "levels_completed": t.get("levels_completed"),
            "raw_reasoning": None,
        })
    return rows


def from_exploration(runs):
    rows = []
    for run in runs:
        folder = EXPLORATION / run
        path = folder / "experiments.jsonl"
        if not path.exists():
            print(f"  [warn] missing exploration run: {path}")
            continue
        frames = folder / "frames"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            exp = json.loads(line)
            eid = exp["experiment_id"]
            before_files = [f"{eid}_step00_start.npy"]
            for s in exp.get("steps", []):
                rows.append({
                    "source": f"exploration/{run}",
                    "episode_id": f"{run}/{eid}",
                    "step": s["step"],
                    "action": s["action"],
                    "action_data": s.get("action_data") or {},
                    "grid_before": load_npy(frames, before_files),
                    "grid_after": load_npy(frames, s.get("after_files")),
                    "before_state": strip_state(s.get("state_before")),
                    "after_state": strip_state(s.get("state_after")),
                    "levels_completed": s.get("levels_completed"),
                    "raw_reasoning": None,
                })
                before_files = s.get("after_files") or []
    return rows


def as_grid(value):
    if value is None:
        return None
    arr = np.array(value)
    return arr[-1] if arr.ndim == 3 else arr


def from_episode_logs():
    rows = []
    for path in sorted((RAW / "logs").glob("*.json")):
        steps = sorted(json.loads(path.read_text()), key=lambda r: r["step"])
        for i, r in enumerate(steps):
            after = r.get("next_grid_state")
            if after is None and i + 1 < len(steps):
                after = steps[i + 1]["grid_state"]
            rows.append({
                "source": f"logs/{path.name}",
                "episode_id": r["episode_id"],
                "step": r["step"],
                "action": r.get("action_name") or f"ACTION{r['action']}",
                "action_data": r.get("action_data") or {},
                "grid_before": as_grid(r.get("grid_state")),
                "grid_after": as_grid(after),
                "before_state": None,
                "after_state": strip_state(r.get("next_state")),
                "levels_completed": None,
                "raw_reasoning": r.get("reasoning") or None,
            })
    return rows


def to_png(grid, name):
    rgb = PALETTE[np.clip(grid.astype(np.int64), 0, 15)]
    img = Image.fromarray(rgb).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": name}


def color_boxes(grid):
    boxes = {}
    for c in np.unique(grid):
        if c == BACKGROUND:
            continue
        ys, xs = np.where(grid == c)
        boxes[int(c)] = (len(xs), int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return boxes


def describe_change(before, after):
    changed = int(np.count_nonzero(before != after))
    if changed == 0:
        return 0, "No pixel changed: the action had no visible effect."

    counter = int(np.count_nonzero(before[:, HUD_COL] != after[:, HUD_COL]))
    floor = int(np.count_nonzero(before[HUD_ROW, :HUD_COL] != after[HUD_ROW, :HUD_COL]))
    before, after = before[:HUD_ROW, :HUD_COL], after[:HUD_ROW, :HUD_COL]
    play = int(np.count_nonzero(before != after))

    parts = []
    if counter:
        parts.append(f"{counter} cell(s) of the move-counter column x={HUD_COL} changed.")
    if floor:
        parts.append(f"{floor} cell(s) of the bottom row y={HUD_ROW} changed.")
    if play == 0:
        parts.append("Nothing changed in the play area.")
        return changed, " ".join(parts)

    ys, xs = np.where(before != after)
    parts.insert(0,
        f"{play} pixels changed in the play area inside the region "
        f"x={xs.min()}..{xs.max()}, y={ys.min()}..{ys.max()}."
    )
    b_boxes, a_boxes = color_boxes(before), color_boxes(after)
    for c in sorted(set(b_boxes) | set(a_boxes)):
        b, a = b_boxes.get(c), a_boxes.get(c)
        if b is None:
            parts.append(f"Colour {c} appeared ({a[0]} px).")
        elif a is None:
            parts.append(f"Colour {c} disappeared ({b[0]} px).")
        elif b == a:
            continue
        elif b[0] == a[0]:
            dx, dy = a[1] - b[1], a[2] - b[2]
            if dx or dy:
                parts.append(f"Colour {c} shape shifted by dx={dx}, dy={dy}.")
            else:
                parts.append(f"Colour {c} pixels rearranged in place.")
        else:
            parts.append(f"Colour {c} pixel count {b[0]} -> {a[0]}.")
    return changed, " ".join(parts)


def transition_hash(before, after, action):
    h = hashlib.sha1()
    h.update(before.astype(np.int8).tobytes())
    h.update(str(action).encode())
    h.update(after.astype(np.int8).tobytes())
    return h.hexdigest()[:16]


def observed_reasoning(row):
    return (
        f"Observation: I executed {action_text(row['action'], json.loads(row['action_data']))}. {row['observation']} "
        f"Game state afterwards: {row['after_state'] or 'unknown'}."
    )


def load_cache():
    cache = {}
    if LISA_CACHE.exists():
        for line in LISA_CACHE.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item["key"]] = item["reasoning"]
    return cache


LISA_DEFAULT_URL = "https://chat-1.ki-awz.iisys.de/api"
LISA_DEFAULT_MODEL = "lisa-pro"


def load_dotenv(path=Path(".env")):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_lisa_client = None


def lisa_client():
    global _lisa_client
    if _lisa_client is None:
        from openai import OpenAI

        if not os.environ.get("LISA_API_KEY"):
            sys.exit("LISA_API_KEY is not set. Add LISA_API_KEY=... to .env.")
        _lisa_client = OpenAI(
            base_url=os.environ.get("LISA_API_URL", LISA_DEFAULT_URL),
            api_key=os.environ["LISA_API_KEY"],
        )
    return _lisa_client


def lisa_model():
    return os.environ.get("LISA_MODEL", LISA_DEFAULT_MODEL)


def lisa_reasoning(row, cache):
    prompt = (
        "You are annotating a dataset for the grid game ARC-AGI-3 AR25.\n"
        f"Action taken: {action_text(row['action'], json.loads(row['action_data']))}\n"
        f"Measured effect (ground truth): {row['observation']}\n"
        f"Game state after: {row['after_state']}\n\n"
        "Write 2-4 sentences of first-person reasoning an agent could have had "
        "before choosing this action, then state what it observed. "
        "Clearly mark anything that is a hypothesis with 'Hypothesis:'. "
        "Do not invent effects that are not in the measured effect."
    )
    key = f"{row['transition_hash']}:{lisa_model()}:{hashlib.sha1(prompt.encode()).hexdigest()[:12]}"
    if key in cache:
        return cache[key]

    response = lisa_client().chat.completions.create(
        model=lisa_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = (response.choices[0].message.content or "").strip()

    cache[key] = text
    with LISA_CACHE.open("a") as f:
        f.write(json.dumps({"key": key, "reasoning": text}) + "\n")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", choices=["observed", "lisa"], default="observed")
    parser.add_argument("--limit", type=int, default=None, help="only first N transitions")
    parser.add_argument("--output", default=str(OUT_DIR / "ar25_transitions.parquet"))
    parser.add_argument(
        "--exploration-runs", nargs="*", default=DEFAULT_EXPLORATION_RUNS,
        help="folders of data/raw/exploration/ to include (pass the flag with no value to include none)",
    )
    args = parser.parse_args()

    if not RAW.exists():
        sys.exit("Run this script from the repository root (data/raw not found).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = (from_sequences() + from_transitions() + from_episode_logs()
                + from_exploration(args.exploration_runs))
    print(f"Loaded {len(raw_rows)} raw transitions")

    rows, skipped = [], 0
    for r in raw_rows:
        b, a = r["grid_before"], r["grid_after"]
        if b is None or a is None or b.shape != (64, 64) or a.shape != (64, 64):
            skipped += 1
            continue
        rows.append(r)
    print(f"Kept {len(rows)} complete transitions, skipped {skipped} incomplete")

    if args.limit:
        rows = rows[: args.limit]

    if args.reasoning == "lisa":
        load_dotenv()
    cache = load_cache() if args.reasoning == "lisa" else {}
    seen = set()
    records = []
    for r in rows:
        b, a = r["grid_before"], r["grid_after"]
        sample_id = f"{r['episode_id']}_step{int(r['step']):03d}"
        changed, observation = describe_change(b, a)
        th = transition_hash(b, a, r["action"])

        rec = {
            "sample_id": sample_id,
            "source": r["source"],
            "episode_id": r["episode_id"],
            "step": int(r["step"]),
            "action": r["action"],
            "action_id": action_id(r["action"]),
            "action_data": json.dumps(r["action_data"]),
            "before_state": r["before_state"],
            "after_state": r["after_state"],
            "levels_completed": r["levels_completed"],
            "changed_pixels": changed,
            "observation": observation,
            "transition_hash": th,
            "is_duplicate": th in seen,
            "raw_reasoning": r["raw_reasoning"],
            "grid_before": b.astype(int).tolist(),
            "grid_after": a.astype(int).tolist(),
            "image_before": to_png(b, f"{sample_id}_before.png"),
            "image_after": to_png(a, f"{sample_id}_after.png"),
        }
        seen.add(th)

        if args.reasoning == "lisa":
            rec["reasoning"] = lisa_reasoning(rec, cache)
            rec["reasoning_source"] = f"lisa:{lisa_model()}"
        else:
            rec["reasoning"] = observed_reasoning(rec)
            rec["reasoning_source"] = "observed_diff"

        rec["instruction"] = INSTRUCTION
        rec["answer"] = f"{rec['reasoning']}\nAction: {action_text(rec['action'], r['action_data'])}"
        records.append(rec)
        print(f"  {sample_id:40s} {r['action']:8s} changed={changed:4d}")

    df = pd.DataFrame(records)
    df.to_parquet(args.output, index=False)

    print(f"\nSaved {len(df)} rows to {args.output}")
    print(f"Duplicates flagged: {int(df['is_duplicate'].sum())}")
    print(df[["sample_id", "action", "changed_pixels", "reasoning"]].head().to_string())


if __name__ == "__main__":
    main()
