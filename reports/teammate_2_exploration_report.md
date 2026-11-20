# AR25 exploration report — Teammate 2

- **Run ID:** `20261120T014608Z`
- **Game:** `ar25-0c556536`, seed 0
- **Raw data:** `data/raw/exploration/20261120T014608Z/` (frames `.npy`, figures `.png`, `experiments.jsonl`, `unsuccessful_experiments.jsonl`)
- **Reproduce:** run `notebooks/teammate_2_exploration.ipynb` top to bottom (random walk seed = 42).

## Environment
```json
{
  "game_id": "ar25-0c556536",
  "seed": 0,
  "frame_count_on_reset": 1,
  "frame_shape": [
    64,
    64
  ],
  "state": "NOT_FINISHED",
  "levels_completed": 0,
  "win_levels": 8,
  "available_actions": [
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7"
  ]
}
```

Background colour: 9. Objects detected on the initial frame: 11.

| object | colour | cells | bbox (x0,y0,x1,y1) | click point |
|---|---|---|---|---|
| obj00 | 10 | 189 | [30, 0, 32, 62] | {'x': 31, 'y': 31} |
| obj01 | 11 | 64 | [63, 0, 63, 63] | {'x': 63, 'y': 31} |
| obj02 | 5 | 40 | [18, 15, 26, 23] | {'x': 23, 'y': 17} |
| obj03 | 4 | 45 | [36, 15, 44, 23] | {'x': 38, 'y': 18} |
| obj04 | 0 | 1 | [19, 16, 19, 16] | {'x': 19, 'y': 16} |
| obj05 | 0 | 1 | [22, 16, 22, 16] | {'x': 22, 'y': 16} |
| obj06 | 0 | 1 | [25, 16, 25, 16] | {'x': 25, 'y': 16} |
| obj07 | 0 | 1 | [25, 19, 25, 19] | {'x': 25, 'y': 19} |
| obj08 | 0 | 1 | [25, 22, 25, 22] | {'x': 25, 'y': 22} |
| obj09 | 11 | 45 | [51, 45, 59, 53] | {'x': 53, 'y': 48} |
| obj10 | 5 | 63 | [0, 63, 62, 63] | {'x': 31, 'y': 63} |

## Experiments (49)
| id | actions | changed cells per step | net change | net bbox | outcome |
|---|---|---|---|---|---|
| E01_ACTION1 | ACTION1 | [109] | 109 | [18, 0, 63, 23] | changed |
| E01_ACTION2 | ACTION2 | [109] | 109 | [18, 0, 63, 26] | changed |
| E01_ACTION3 | ACTION3 | [109] | 109 | [15, 0, 63, 23] | changed |
| E01_ACTION4 | ACTION4 | [109] | 109 | [18, 0, 63, 23] | changed |
| E01_ACTION5 | ACTION5 | [1] | 1 | [63, 0, 63, 0] | changed_then_returned_to_start |
| E01_ACTION7 | ACTION7 | [0] | 0 | None | no_visible_change |
| E02_ACTION1_x3 | ACTION1, ACTION1, ACTION1 | [109, 109, 109] | 183 | [18, 0, 63, 23] | changed |
| E02_ACTION2_x3 | ACTION2, ACTION2, ACTION2 | [109, 109, 109] | 183 | [18, 0, 63, 32] | changed |
| E02_ACTION3_x3 | ACTION3, ACTION3, ACTION3 | [109, 109, 109] | 183 | [9, 0, 63, 23] | changed |
| E02_ACTION4_x3 | ACTION4, ACTION4, ACTION4 | [109, 79, 80] | 155 | [18, 0, 63, 23] | changed |
| E02_ACTION5_x3 | ACTION5, ACTION5, ACTION5 | [1, 1, 1] | 3 | [63, 0, 63, 2] | changed_then_returned_to_start |
| E02_ACTION7_x3 | ACTION7, ACTION7, ACTION7 | [0, 0, 0] | 0 | None | no_visible_change |
| E03_ACTION1_then_ACTION2 | ACTION1, ACTION2 | [109, 109] | 2 | [63, 0, 63, 1] | changed_then_returned_to_start |
| E03_ACTION2_then_ACTION1 | ACTION2, ACTION1 | [109, 109] | 2 | [63, 0, 63, 1] | changed_then_returned_to_start |
| E03_ACTION3_then_ACTION4 | ACTION3, ACTION4 | [109, 109] | 2 | [63, 0, 63, 1] | changed_then_returned_to_start |
| E03_ACTION4_then_ACTION3 | ACTION4, ACTION3 | [109, 109] | 2 | [63, 0, 63, 1] | changed_then_returned_to_start |
| E04_click_obj00_c10 | ACTION6{'x': 31, 'y': 31} | [0] | 0 | None | no_visible_change |
| E04_click_obj01_c11 | ACTION6{'x': 63, 'y': 31} | [0] | 0 | None | no_visible_change |
| E04_click_obj02_c5 | ACTION6{'x': 23, 'y': 17} | [0] | 0 | None | no_visible_change |
| E04_click_obj03_c4 | ACTION6{'x': 38, 'y': 18} | [0] | 0 | None | no_visible_change |
| E04_click_obj04_c0 | ACTION6{'x': 19, 'y': 16} | [0] | 0 | None | no_visible_change |
| E04_click_obj05_c0 | ACTION6{'x': 22, 'y': 16} | [0] | 0 | None | no_visible_change |
| E04_click_obj06_c0 | ACTION6{'x': 25, 'y': 16} | [0] | 0 | None | no_visible_change |
| E04_click_obj07_c0 | ACTION6{'x': 25, 'y': 19} | [0] | 0 | None | no_visible_change |
| E04_click_obj08_c0 | ACTION6{'x': 25, 'y': 22} | [0] | 0 | None | no_visible_change |
| E04_click_obj09_c11 | ACTION6{'x': 53, 'y': 48} | [0] | 0 | None | no_visible_change |
| E04_click_obj10_c5 | ACTION6{'x': 31, 'y': 63} | [0] | 0 | None | no_visible_change |
| E04_click_background_0 | ACTION6{'x': 5, 'y': 5} | [0] | 0 | None | no_visible_change |
| E04_click_background_1 | ACTION6{'x': 26, 'y': 56} | [0] | 0 | None | no_visible_change |
| E04_click_background_2 | ACTION6{'x': 56, 'y': 32} | [0] | 0 | None | no_visible_change |
| E05_ACTION1_run1 | ACTION1, ACTION1 | [109, 109] | 146 | [18, 0, 63, 23] | changed |
| E05_ACTION1_run2 | ACTION1, ACTION1 | [109, 109] | 146 | [18, 0, 63, 23] | changed |
| E05_ACTION2_run1 | ACTION2, ACTION2 | [109, 109] | 146 | [18, 0, 63, 29] | changed |
| E05_ACTION2_run2 | ACTION2, ACTION2 | [109, 109] | 146 | [18, 0, 63, 29] | changed |
| E05_ACTION3_run1 | ACTION3, ACTION3 | [109, 109] | 146 | [12, 0, 63, 23] | changed |
| E05_ACTION3_run2 | ACTION3, ACTION3 | [109, 109] | 146 | [12, 0, 63, 23] | changed |
| E05_ACTION4_run1 | ACTION4, ACTION4 | [109, 79] | 116 | [18, 0, 63, 23] | changed |
| E05_ACTION4_run2 | ACTION4, ACTION4 | [109, 79] | 116 | [18, 0, 63, 23] | changed |
| E06_click_obj00_then_ACTION1 | ACTION6{'x': 31, 'y': 31}, ACTION1 | [0, 109] | 109 | [18, 0, 63, 23] | changed |
| E06_click_obj01_then_ACTION1 | ACTION6{'x': 63, 'y': 31}, ACTION1 | [0, 109] | 109 | [18, 0, 63, 23] | changed |
| E06_click_obj10_then_ACTION1 | ACTION6{'x': 31, 'y': 63}, ACTION1 | [0, 109] | 109 | [18, 0, 63, 23] | changed |
| E06_click_obj03_then_ACTION1 | ACTION6{'x': 38, 'y': 18}, ACTION1 | [0, 109] | 109 | [18, 0, 63, 23] | changed |
| E07_random_walk_30 | ACTION1, ACTION6{'x': 41, 'y': 28}, ACTION4, ACTION7, … (30 total) | [109, 0, 109, 108, 109, 1] | 124 | [18, 0, 63, 23] | changed |
| E08_ACTION1_then_ACTION7 | ACTION1, ACTION7 | [109, 108] | 1 | [63, 0, 63, 0] | changed_then_returned_to_start |
| E08_ACTION3_then_ACTION7 | ACTION3, ACTION7 | [109, 108] | 1 | [63, 0, 63, 0] | changed_then_returned_to_start |
| E08_ACTION1_ACTION1_then_ACTION7_x2 | ACTION1, ACTION1, ACTION7, ACTION7 | [109, 109, 108, 108] | 2 | [63, 0, 63, 1] | changed_then_returned_to_start |
| E08_ACTION5_then_ACTION7 | ACTION5, ACTION7 | [1, 0] | 1 | [63, 0, 63, 0] | changed_then_returned_to_start |
| E09_yellow_onto_static_L | ACTION2, ACTION2, ACTION2, ACTION2, … (15 total) | [109, 109, 109, 109, 109, 109] | 685 | [3, 0, 59, 62] | level_progress |
| E09b_yellow_onto_static_L_horizontal_first | ACTION3, ACTION3, ACTION3, ACTION3, … (15 total) | [109, 109, 109, 109, 109, 109] | 685 | [3, 0, 59, 62] | level_progress |

## Observations (measured)
- E01_ACTION1: ACTION1  changed 109 cells in bbox [18, 0, 63, 23]; colour transitions {'9->4': 27, '4->9': 27, '9->5': 24, '5->9': 24}
- E01_ACTION2: ACTION2  changed 109 cells in bbox [18, 0, 63, 26]; colour transitions {'4->9': 27, '9->4': 27, '5->9': 24, '9->5': 24}
- E01_ACTION3: ACTION3  changed 109 cells in bbox [15, 0, 63, 23]; colour transitions {'4->9': 27, '9->4': 27, '9->5': 24, '5->9': 24}
- E01_ACTION4: ACTION4  changed 109 cells in bbox [18, 0, 63, 23]; colour transitions {'9->4': 27, '4->9': 27, '5->9': 24, '9->5': 24}
- E01_ACTION5: ACTION5  changed 1 cells in bbox [63, 0, 63, 0]; colour transitions {'11->5': 1}
- E01_ACTION7: ACTION7  changed 0 cells in bbox None; colour transitions {}
- E04_click_obj00_c10: ACTION6 {'x': 31, 'y': 31} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj01_c11: ACTION6 {'x': 63, 'y': 31} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj02_c5: ACTION6 {'x': 23, 'y': 17} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj03_c4: ACTION6 {'x': 38, 'y': 18} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj04_c0: ACTION6 {'x': 19, 'y': 16} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj05_c0: ACTION6 {'x': 22, 'y': 16} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj06_c0: ACTION6 {'x': 25, 'y': 16} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj07_c0: ACTION6 {'x': 25, 'y': 19} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj08_c0: ACTION6 {'x': 25, 'y': 22} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj09_c11: ACTION6 {'x': 53, 'y': 48} changed 0 cells in bbox None; colour transitions {}
- E04_click_obj10_c5: ACTION6 {'x': 31, 'y': 63} changed 0 cells in bbox None; colour transitions {}
- E04_click_background_0: ACTION6 {'x': 5, 'y': 5} changed 0 cells in bbox None; colour transitions {}
- E04_click_background_1: ACTION6 {'x': 26, 'y': 56} changed 0 cells in bbox None; colour transitions {}
- E04_click_background_2: ACTION6 {'x': 56, 'y': 32} changed 0 cells in bbox None; colour transitions {}
- Reset is deterministic: True
- Same sequence → same final frame: {'ACTION1': True, 'ACTION2': True, 'ACTION3': True, 'ACTION4': True}

## Hypotheses
| id | hypothesis | status | evidence |
|---|---|---|---|
| H1_ACTION1_ACTION2 | ACTION2 brings the objects back after ACTION1 (counter column ignored) | supported | E03_ACTION1_then_ACTION2: outcome=changed_then_returned_to_start, net change=2; objects net change=0 |
| H1_ACTION2_ACTION1 | ACTION1 brings the objects back after ACTION2 (counter column ignored) | supported | E03_ACTION2_then_ACTION1: outcome=changed_then_returned_to_start, net change=2; objects net change=0 |
| H1_ACTION3_ACTION4 | ACTION4 brings the objects back after ACTION3 (counter column ignored) | supported | E03_ACTION3_then_ACTION4: outcome=changed_then_returned_to_start, net change=2; objects net change=0 |
| H1_ACTION4_ACTION3 | ACTION3 brings the objects back after ACTION4 (counter column ignored) | supported | E03_ACTION4_then_ACTION3: outcome=changed_then_returned_to_start, net change=2; objects net change=0 |
| H2 | Clicking empty background has no visible effect | supported | E04_click_background_0: outcome=no_visible_change, net change=0; E04_click_background_1: outcome=no_visible_change, net change=0; E04_click_background_2: outcome=no_visible_change, net change=0 |
| H3 | The game is deterministic for a fixed seed and action sequence | supported | E05 identical finals: {'ACTION1': True, 'ACTION2': True, 'ACTION3': True, 'ACTION4': True}; reset identical: True |
| H4_ACTION1 | Repeating ACTION1 keeps changing the frame (cumulative effect) | supported | E02_ACTION1_x3: changed cells per step [109, 109, 109] |
| H4_ACTION2 | Repeating ACTION2 keeps changing the frame (cumulative effect) | supported | E02_ACTION2_x3: changed cells per step [109, 109, 109] |
| H4_ACTION3 | Repeating ACTION3 keeps changing the frame (cumulative effect) | supported | E02_ACTION3_x3: changed cells per step [109, 109, 109] |
| H4_ACTION4 | Repeating ACTION4 keeps changing the frame (cumulative effect) | supported | E02_ACTION4_x3: changed cells per step [109, 79, 80] |
| H4_ACTION5 | Repeating ACTION5 keeps changing the frame (cumulative effect) | supported | E02_ACTION5_x3: changed cells per step [1, 1, 1] |
| H4_ACTION7 | Repeating ACTION7 keeps changing the frame (cumulative effect) | inconclusive | E02_ACTION7_x3: changed cells per step [0, 0, 0] |
| H5 | ACTION1/ACTION2 move both the dotted L and the yellow L up/down by 3 cells | supported | run 20261120T012831Z, E01: both bboxes y 16→13 (ACTION1) and 16→19 (ACTION2); x unchanged |
| H5b | ACTION3/ACTION4 move the two Ls horizontally by 3 cells in opposite directions (mirror around the white bar); ACTION3 = dotted L left / yellow L right | supported | run 20261120T012831Z, E01: ACTION3 dotted x 19→16, yellow x 36→39; ACTION4 the reverse |
| H5c | The yellow L is the mirror image of the dotted L across the white bar (x=30–32) | supported | run 20261120T012831Z, initial frame: dotted x18–26, yellow x36–44 (symmetric around x=31), shapes mirrored |
| H5d | The Ls can cross the white bar (they are not blocked by it) | supported | run 20261120T012831Z, E02_ACTION4_x3 step 3: dotted L at x26–34, yellow L at x27–29, overlapping the bar |
| H6 | Clicking an object selects it, and movement actions then apply to the selected object | refuted | run 20261120T012831Z, E04: no click changed anything; E06: ACTION1 after any click = same 109-cell change as E01 |
| H7 | Placing the yellow L exactly on the static L (colour 11) completes level 1, whatever the move order | supported | E09_yellow_onto_static_L: outcome=level_progress, level reached at step 15/15; E09b_yellow_onto_static_L_horizontal_first: outcome=level_progress, level reached at step 15/15 |
| H8 | ACTION7 undoes the previous move (one move per ACTION7) | supported | E08_ACTION1_then_ACTION7: objects net change=0; E08_ACTION3_then_ACTION7: objects net change=0; E08_ACTION1_ACTION1_then_ACTION7_x2: objects net change=0 |
| H8b | ACTION7 gives back the counter cell of the undone move | refuted | counter after each step: {'E08_ACTION1_then_ACTION7': [1, 1], 'E08_ACTION3_then_ACTION7': [1, 1], 'E08_ACTION1_ACTION1_then_ACTION7_x2': [1, 2, 2, 2]} |
| H9 | Column x=63 is a move counter / budget: one cell turns 11→5 per action (ACTION1–5); clicks do not count | supported | run 20261120T012831Z: E01 ACTION1–5 each change exactly 1 cell in column 63; E04 clicks change 0 |

## Unsuccessful experiments (16)
- `E01_ACTION7` — no_visible_change
- `E02_ACTION7_x3` — no_visible_change
- `E04_click_obj00_c10` — no_visible_change
- `E04_click_obj01_c11` — no_visible_change
- `E04_click_obj02_c5` — no_visible_change
- `E04_click_obj03_c4` — no_visible_change
- `E04_click_obj04_c0` — no_visible_change
- `E04_click_obj05_c0` — no_visible_change
- `E04_click_obj06_c0` — no_visible_change
- `E04_click_obj07_c0` — no_visible_change
- `E04_click_obj08_c0` — no_visible_change
- `E04_click_obj09_c11` — no_visible_change
- `E04_click_obj10_c5` — no_visible_change
- `E04_click_background_0` — no_visible_change
- `E04_click_background_1` — no_visible_change
- `E04_click_background_2` — no_visible_change

## Discoveries
- Reset and action sequences are deterministic (E05).
- Two objects are controlled together: the dotted L (left half) and the yellow L (right half), which mirror each other across the white bar.
- ACTION1 = up 3 cells, ACTION2 = down 3 cells for both Ls; ACTION3 = dotted L left / yellow L right, ACTION4 = the opposite (E01, E02).
- The Ls pass through the white bar and can swap sides (E02_ACTION4_x3).
- Column x=63 fills one cell per move: it behaves like a move counter (E01, E03). It resets to empty when a level is completed (E09).
- ACTION5 only advances the counter: no other visible effect (E01_ACTION5, E02_ACTION5_x3).
- ACTION6 (click) never changed anything on level 1, on objects or background, and does not advance the counter (E04, E06).
- ACTION7 undoes the last move (E08). It does not advance the counter, and does not give the counter cell back.
- LEVEL 1 SOLVED: moving the yellow L exactly onto the static L of the same shape completes level 1 in 15 moves (10 x ACTION2 + 5 x ACTION3, in either order) — levels_completed 0 → 1, 3-frame transition, new level shown (E09, E09b).
- 30 seeded random actions did not complete a level nor end the game (E07).

## Open questions
- What does level 2 look like, and does the same rule (move the mirrored shape onto the static target) apply?
- What happens when the counter column is full (64 moves)? Game over or level reset?
- Does the dotted L have any role besides mirroring the yellow L (e.g. obstacles in later levels)?
- Is ACTION5 useful later (e.g. a 'wait' action), and does ACTION6 do anything in later levels?
- Level-1 human baseline is 32 actions (env.info.baseline_actions); our 15-move solution is shorter — confirm what the scoring counts.
