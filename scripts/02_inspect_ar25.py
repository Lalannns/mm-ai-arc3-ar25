from arc_agi import Arcade

GAME_ID = "ar25-0c556536"

arc = Arcade()
env = arc.make(
    game_id=GAME_ID,
    seed=0,
    save_recording=False,
    include_frame_data=True,
)

if env is None:
    raise RuntimeError("Could not create AR25 environment")

print("=== Environment ===")
print("Type:", type(env))
print("Info:", env.info)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

print("\n=== Initial observation ===")
obs = env.reset()

print("Type:", type(obs))

if obs is not None:
    print("Attributes:", [x for x in dir(obs) if not x.startswith("_")])

    for name in ["frame", "state", "reward", "status", "levels_completed"]:
        if hasattr(obs, name):
            value = getattr(obs, name)
            if name == "frame" and isinstance(value, list):
                print(f"{name}: list with {len(value)} item(s)")
                for i, item in enumerate(value):
                    print(f"  frame[{i}] type={type(item)}")
                    if hasattr(item, "shape"):
                        print(f"  frame[{i}] shape={item.shape}")
            else:
                print(f"{name}: {value}")
