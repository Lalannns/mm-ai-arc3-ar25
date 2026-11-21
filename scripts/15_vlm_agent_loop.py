import argparse
import base64
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

GAME_ID = "ar25-0c556536"
DEFAULT_MODEL = os.environ.get("AR25_VLM_MODEL", "gpt-4o-mini")
OUTPUT = Path("data/raw/trajectories")
FRAME_OUTPUT = OUTPUT / "sequence_frames"
TERMINAL_STATES = {"WIN", "GAME_OVER"}

PALETTE = np.array(
    [
        (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
        (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
        (127, 219, 255), (135, 12, 37), (255, 255, 255), (60, 60, 60),
        (0, 128, 128), (128, 0, 128), (128, 128, 0), (200, 120, 80),
    ],
    dtype=np.uint8,
)


def state_name(obs) -> str:
    val = getattr(obs.state, "value", obs.state)
    return str(val).replace("GameState.", "")


def grid_hash(grid: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()[:12]


def save_frames(frames, trajectory_id: str, step: int, label: str) -> list[str]:
    filenames = []
    for idx, frame in enumerate(frames):
        name = f"{trajectory_id}_step{step:03d}_{label}_{idx:02d}.npy"
        np.save(FRAME_OUTPUT / name, frame)
        filenames.append(name)
    return filenames


def grid_to_png_b64(grid: np.ndarray, size: int = 512) -> str:
    clipped = np.clip(np.asarray(grid, dtype=np.int64), 0, 15)
    rgb = PALETTE[clipped]
    img = Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def summarize_diff(before: np.ndarray, after: np.ndarray) -> str:
    if before.shape != after.shape:
        return f"shape changed {before.shape} -> {after.shape}"
    mask = before != after
    count = int(np.count_nonzero(mask))
    if count == 0:
        return "0 pixels changed (no visible change)"
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    c_before = Counter(before.flatten().tolist())
    c_after = Counter(after.flatten().tolist())
    deltas = []
    for col in sorted(set(c_before) | set(c_after)):
        if c_before[col] != c_after[col]:
            deltas.append(f"c{col}:{c_before[col]}->{c_after[col]}")
    delta_str = ", ".join(deltas[:8]) if deltas else "counts unchanged"
    return f"{count} px changed in x={x0}..{x1}, y={y0}..{y1} ({delta_str})"


def available_action_names(obs, fallback: list[str]) -> list[str]:
    raw_avail = getattr(obs, "available_actions", None) or []
    names = []
    for item in raw_avail:
        if isinstance(item, int):
            candidate = f"ACTION{item}"
            if candidate in GameAction.__members__ and candidate != "RESET":
                names.append(candidate)
        else:
            s = str(item).replace("GameAction.", "")
            if s in GameAction.__members__ and s != "RESET":
                names.append(s)
    return names or fallback


def choose_vlm_action(
    client: OpenAI,
    model: str,
    grid: np.ndarray,
    state_str: str,
    levels_completed: int,
    step_num: int,
    max_steps: int,
    valid_actions: list[str],
    history: list[dict],
) -> tuple[str, dict, str]:
    png_b64 = grid_to_png_b64(grid)
    recent_txt = "\n".join(
        f"- step {h['step']}: {h['action']} data={h['action_data']} -> {h['diff']}"
        for h in history[-8:]
    ) or "- none yet"

    system_prompt = (
        "You are an autonomous agent playing ARC-AGI-3 game ar25-0c556536 on a 64x64 grid. "
        "Infer mechanics purely from visual observations and recent transition history. "
        "Never guess hidden internals; propose a concise hypothesis and pick one valid action. "
        "Respond ONLY with valid JSON: "
        '{"reasoning": "...", "action": "ACTION1", "action_data": {}} '
        "For ACTION6, include {\"x\": int(0..63), \"y\": int(0..63)} in action_data; otherwise {}."
    )
    user_prompt = (
        f"Step {step_num}/{max_steps} | State: {state_str} | Levels completed: {levels_completed}\n"
        f"Grid shape: {grid.shape} | State hash: {grid_hash(grid)}\n"
        f"Valid actions: {valid_actions}\n"
        f"Recent transitions:\n{recent_txt}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}", "detail": "high"},
                    },
                ],
            },
        ],
    )
    raw_content = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw_content)
    action = str(parsed.get("action", "")).strip()
    if action not in valid_actions:
        action = valid_actions[0]
    data = parsed.get("action_data")
    if not isinstance(data, dict):
        data = {}
    if action == "ACTION6":
        x = int(np.clip(int(data.get("x", 32)), 0, grid.shape[1] - 1))
        y = int(np.clip(int(data.get("y", 32)), 0, grid.shape[0] - 1))
        data = {"x": x, "y": y}
    else:
        data = {}
    reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."
    return action, data, reasoning


def pick_fallback_action(
    grid: np.ndarray,
    valid_actions: list[str],
    tried_on_state: set[tuple[str, str]],
    step_num: int,
) -> tuple[str, dict, str]:
    gh = grid_hash(grid)
    for act in valid_actions:
        if (gh, act) not in tried_on_state:
            data = {"x": 32, "y": 32} if act == "ACTION6" else {}
            return act, data, f"Fallback exploration on state {gh}: trying untried {act}."
    act = valid_actions[step_num % len(valid_actions)]
    data = {"x": 32, "y": 32} if act == "ACTION6" else {}
    return act, data, f"Fallback cycle on state {gh}: all actions tried, executing {act}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous VLM agent loop for AR25")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-no-change", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic fallback instead of VLM API")
    args = parser.parse_args()

    load_dotenv()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FRAME_OUTPUT.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set. Export it or pass --dry-run.")
        client = OpenAI()

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trajectory_id = f"{run_stamp}_vlm_s{args.seed}"

    arc = Arcade()
    env = arc.make(
        game_id=GAME_ID,
        seed=args.seed,
        save_recording=False,
        include_frame_data=True,
    )
    if env is None:
        raise RuntimeError(f"Could not initialize {GAME_ID}")

    current = env.reset()
    if current is None:
        raise RuntimeError("Environment reset returned None")

    initial_files = save_frames(current.frame, trajectory_id, 0, "initial")
    default_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]

    records = []
    history = []
    tried_on_state: set[tuple[str, str]] = set()
    consecutive_no_change = 0
    initial_levels = int(getattr(current, "levels_completed", 0) or 0)

    for step_num in range(1, args.max_steps + 1):
        curr_state = state_name(current)
        if curr_state in TERMINAL_STATES:
            print(f"Terminal state reached before step {step_num}: {curr_state}")
            break

        grid_now = np.asarray(current.frame[-1], dtype=np.int64)
        gh_now = grid_hash(grid_now)
        valid_actions = available_action_names(current, default_actions)

        if args.dry_run:
            action_name, action_data, reasoning = pick_fallback_action(
                grid_now, valid_actions, tried_on_state, step_num
            )
        else:
            try:
                action_name, action_data, reasoning = choose_vlm_action(
                    client=client,
                    model=args.model,
                    grid=grid_now,
                    state_str=curr_state,
                    levels_completed=int(getattr(current, "levels_completed", 0) or 0),
                    step_num=step_num,
                    max_steps=args.max_steps,
                    valid_actions=valid_actions,
                    history=history,
                )
                if (gh_now, action_name) in tried_on_state and consecutive_no_change >= 2:
                    action_name, action_data, fb_reason = pick_fallback_action(
                        grid_now, valid_actions, tried_on_state, step_num
                    )
                    reasoning = f"{reasoning} | {fb_reason}"
            except Exception as exc:
                action_name, action_data, fb_reason = pick_fallback_action(
                    grid_now, valid_actions, tried_on_state, step_num
                )
                reasoning = f"VLM fallback ({exc}): {fb_reason}"

        tried_on_state.add((gh_now, action_name))
        before_files = save_frames(current.frame, trajectory_id, step_num, "before")
        before_frames = current.frame

        result = env.step(GameAction[action_name], data=action_data)
        if result is None:
            raise RuntimeError(f"{action_name} returned None at step {step_num}")

        after_files = save_frames(result.frame, trajectory_id, step_num, "after")
        changed_pixels = [
            None if b.shape != a.shape else int(np.count_nonzero(b != a))
            for b, a in zip(before_frames, result.frame)
        ]
        total_changed = sum(c for c in changed_pixels if c is not None)
        diff_summary = summarize_diff(
            np.asarray(before_frames[-1], dtype=np.int64),
            np.asarray(result.frame[-1], dtype=np.int64),
        )

        if total_changed == 0:
            consecutive_no_change += 1
        else:
            consecutive_no_change = 0

        after_state = state_name(result)
        levels_now = int(getattr(result, "levels_completed", 0) or 0)

        record = {
            "trajectory_id": trajectory_id,
            "step": step_num,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "game_id": GAME_ID,
            "seed": args.seed,
            "action": action_name,
            "action_data": action_data,
            "before_files": before_files,
            "after_files": after_files,
            "before_frame_count": len(before_frames),
            "after_frame_count": len(result.frame),
            "changed_pixels_per_frame": changed_pixels,
            "before_state": curr_state,
            "after_state": after_state,
            "levels_completed": levels_now,
            "win_levels": getattr(result, "win_levels", None),
            "available_actions": getattr(result, "available_actions", None),
            "full_reset": getattr(result, "full_reset", None),
            "action_input": {
                "reasoning": reasoning,
                "action_data": action_data,
                "model": "dry_run_fallback" if args.dry_run else args.model,
            },
        }
        records.append(record)
        history.append({
            "step": step_num,
            "action": action_name,
            "action_data": action_data,
            "diff": diff_summary,
        })

        print(
            f"Step {step_num:02d}: {action_name} {action_data or ''} | "
            f"{curr_state} -> {after_state} | {diff_summary}"
        )

        current = result
        if levels_now > initial_levels:
            print(f"Level completed! levels_completed={levels_now}")
        if after_state in TERMINAL_STATES:
            print(f"Stopping at terminal state: {after_state}")
            break
        if consecutive_no_change >= args.max_no_change:
            print(f"Stopping after {consecutive_no_change} consecutive no-change steps")
            break

    final_state = state_name(current)
    trajectory_record = {
        "trajectory_id": trajectory_id,
        "game_id": GAME_ID,
        "seed": args.seed,
        "initial_files": initial_files,
        "step_count": len(records),
        "final_state": final_state,
        "steps": records,
    }

    jsonl_path = OUTPUT / "sequences.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(trajectory_record) + "\n")

    print(
        f"\nSaved {len(records)} steps ({trajectory_id}) -> {jsonl_path} | "
        f"final_state={final_state} | levels_completed={getattr(current, 'levels_completed', 0)}"
    )


if __name__ == "__main__":
    main()
