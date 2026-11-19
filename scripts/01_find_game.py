from arc_agi import Arcade

arc = Arcade()

games = arc.get_environments()

print(f"Found {len(games)} environments\n")

for game in games:
    print(game)
