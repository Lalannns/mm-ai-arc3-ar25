# AR25 level 2 exploration report — Teammate 2

- **Run ID:** `20261120T020820Z`
- **Raw data:** `data/raw/exploration/level2_20261120T020820Z/`
- **How level 2 is reached:** reset + level-1 solution (10 × ACTION2, 5 × ACTION3) from `teammate_2_exploration.ipynb`.
- **Reset behaviour:** [{'levels_after_reset': 0, 'full_reset': True, 'replayed': True, 'identical_to_first_level2_frame': True}, {'levels_after_reset': 0, 'full_reset': True, 'replayed': True, 'identical_to_first_level2_frame': True}]

## Level-2 start positions
```json
{
 "yellow": {
  "x0": 15,
  "y0": 18,
  "x1": 29,
  "y1": 29,
  "cells": 72
 },
 "grey": {
  "x0": 45,
  "y0": 18,
  "x1": 59,
  "y1": 29,
  "cells": 64
 },
 "bar": {
  "x0": 36,
  "y0": 0,
  "x1": 38,
  "y1": 62,
  "cells": 168
 },
 "target": {
  "x0": 3,
  "y0": 42,
  "x1": 17,
  "y1": 53,
  "cells": 72
 }
}
```

## Measured displacement per action (dx, dy of each object's top-left corner)
| action | displacement |
|---|---|
| ACTION1 | {'yellow': (0, 0), 'grey': (0, 0), 'bar': (0, 0), 'target': (0, 0)} |
| ACTION2 | {'yellow': (0, 0), 'grey': (0, 0), 'bar': (0, 0), 'target': (0, 0)} |
| ACTION3 | {'yellow': (-6, 0), 'grey': (0, 0), 'bar': (-3, 0), 'target': (0, 0)} |
| ACTION4 | {'yellow': (6, 0), 'grey': (0, 0), 'bar': (3, 0), 'target': (0, 0)} |
| ACTION5 | {'yellow': (0, 0), 'grey': (0, 0), 'bar': (0, 0), 'target': (0, 0)} |
| ACTION7 | {'yellow': (0, 0), 'grey': (0, 0), 'bar': (0, 0), 'target': (0, 0)} |

## Experiments (26)
| id | actions | changed cells per step | objects net change | outcome |
|---|---|---|---|---|
| L2_E01_ACTION1 | ACTION1 | [0] | 0 | no_visible_change |
| L2_E01_ACTION2 | ACTION2 | [0] | 0 | no_visible_change |
| L2_E01_ACTION3 | ACTION3 | [487] | 486 | changed |
| L2_E01_ACTION4 | ACTION4 | [487] | 486 | changed |
| L2_E01_ACTION5 | ACTION5 | [30] | 29 | changed |
| L2_E01_ACTION7 | ACTION7 | [0] | 0 | no_visible_change |
| L2_E02_ACTION1_x5 | ACTION1, ACTION1, ACTION1, ACTION1, … (5 total) | [0, 0, 0, 0, 0] | 0 | no_visible_change |
| L2_E02_ACTION2_x5 | ACTION2, ACTION2, ACTION2, ACTION2, … (5 total) | [0, 0, 0, 0, 0] | 0 | no_visible_change |
| L2_E02_ACTION3_x5 | ACTION3, ACTION3, ACTION3, ACTION3, … (5 total) | [487, 487, 478, 442, 397] | 414 | changed |
| L2_E02_ACTION4_x5 | ACTION4, ACTION4, ACTION4, ACTION4, … (5 total) | [487, 478, 461, 426, 393] | 454 | changed |
| L2_E03_ACTION5_then_ACTION1 | ACTION5, ACTION1 | [30, 181] | 199 | changed |
| L2_E03_ACTION5_then_ACTION3 | ACTION5, ACTION3 | [30, 145] | 165 | changed |
| L2_E04_click_yellow | ACTION6{'x': 21, 'y': 18} | [0] | 0 | no_visible_change |
| L2_E04_click_grey | ACTION6{'x': 45, 'y': 18} | [29] | 29 | changed |
| L2_E04_click_bar | ACTION6{'x': 36, 'y': 0} | [0] | 0 | no_visible_change |
| L2_E04_click_target | ACTION6{'x': 9, 'y': 42} | [0] | 0 | no_visible_change |
| L2_E04_click_bar_dot | ACTION6{'x': 37, 'y': 1} | [0] | 0 | no_visible_change |
| L2_E04_click_background | ACTION6{'x': 5, 'y': 5} | [0] | 0 | no_visible_change |
| L2_E04_click_bar_then_ACTION3 | ACTION6{'x': 36, 'y': 0}, ACTION3 | [0, 487] | 486 | changed |
| L2_E04_click_bar_then_ACTION4 | ACTION6{'x': 36, 'y': 0}, ACTION4 | [0, 487] | 486 | changed |
| L2_E06_ACTION5_x1 | ACTION5 | [30] | 29 | changed |
| L2_E06_ACTION5_x2 | ACTION5, ACTION5 | [30, 30] | 0 | changed_then_returned_to_start |
| L2_E06_ACTION5_x3 | ACTION5, ACTION5, ACTION5 | [30, 30, 30] | 29 | changed |
| L2_E06_click_grey_then_ACTION2 | ACTION6{'x': 45, 'y': 18}, ACTION2 | [29, 181] | 199 | changed |
| L2_E07_solve_with_ACTION5 | ACTION3, ACTION3, ACTION5, ACTION2, … (11 total) | [487, 487, 30, 181, 181, 181] | 784 | level_progress |
| L2_E07b_solve_with_click | ACTION3, ACTION3, ACTION6{'x': 45, 'y': 18}, ACTION2, … (11 total) | [487, 487, 29, 181, 181, 181] | 784 | level_progress |

## Hypotheses
| id | hypothesis | status | evidence |
|---|---|---|---|
| H_L2_0 | At the start of level 2 (bar selected), ACTION1/ACTION2 still move the yellow shape up/down by 3 cells | refuted | L2_E01 mapping: {'ACTION1': {'yellow': (0, 0), 'grey': (0, 0), 'bar': (0, 0), 'target': (0, 0)}, 'ACTION2': {'yellow': (0, 0), 'grey': (0, 0), 'bar': (0, 0), 'target': (0, 0)}} |
| H_L2_mirror | At the start of level 2 (bar selected), ACTION3/ACTION4 still move the yellow and grey shapes in opposite horizontal directions | refuted | L2_E01 mapping: {'ACTION3': {'yellow': (-6, 0), 'grey': (0, 0), 'bar': (-3, 0), 'target': (0, 0)}, 'ACTION4': {'yellow': (6, 0), 'grey': (0, 0), 'bar': (3, 0), 'target': (0, 0)}} |
| H_L2_click | Clicks have no effect on level 2 either | refuted | clicks with an effect: ['L2_E04_click_grey'] |
| H_L2_1 | Moving the yellow shape onto the target completes level 2 (same rule as level 1) | untested | no direct plan from the measured mapping; remainder=(0, 24) |
| H_L2_bar | At the start of level 2, ACTION3/ACTION4 move the bar (mirror axis) horizontally | supported | L2_E01: {'ACTION3': (-3, 0), 'ACTION4': (3, 0)} |
| H_L2_axis | The yellow shape is the mirror image of the grey shape across the bar: moving the bar by d moves yellow by 2d | supported | L2_E01 yellow vs bar: {'ACTION3': ((-6, 0), (-3, 0)), 'ACTION4': ((6, 0), (3, 0))} |
| H_L2_select | The black dots mark the selected object; ACTION5 toggles the selection between bar and grey shape | supported | dots on: {'start': 'bar', 'L2_E06_ACTION5_x1': 'grey', 'L2_E06_ACTION5_x2': 'bar', 'L2_E06_ACTION5_x3': 'grey', 'after click on grey': 'grey'} |
| H_L2_click_select | Clicking the grey shape selects it (without using a counter cell) | supported | dots after click: grey; counter cells used by the click: 0 |
| H_L2_solve | Bar to align horizontally, then grey shape down, completes level 2 | supported | L2_E07_solve_with_ACTION5: level_progress in 11 actions; L2_E07b_solve_with_click: level_progress in 11 actions |

## Unsuccessful experiments (10)
- `L2_E01_ACTION1` — no_visible_change
- `L2_E01_ACTION2` — no_visible_change
- `L2_E01_ACTION7` — no_visible_change
- `L2_E02_ACTION1_x5` — no_visible_change
- `L2_E02_ACTION2_x5` — no_visible_change
- `L2_E04_click_yellow` — no_visible_change
- `L2_E04_click_bar` — no_visible_change
- `L2_E04_click_target` — no_visible_change
- `L2_E04_click_bar_dot` — no_visible_change
- `L2_E04_click_background` — no_visible_change

## Discoveries
- RESET from level 2 goes back to level 1 (full_reset=True, levels_completed=0); level 2 is reached again by replaying the 15 level-1 moves (goto_level2 log).
- The black dots (colour 0) mark the selected object. At the start of level 2 the bar is selected (run 20261120T015752Z).
- With the bar selected: ACTION3/ACTION4 move the bar 3 cells left/right and the yellow shape 6 cells the same way; the grey shape does not move. ACTION1/ACTION2 do nothing and do not use a counter cell (L2_E01).
- ACTION5 moves the selection to the grey shape (21 dots leave the bar, 8 appear on the grey shape) and uses one counter cell (L2_E01_ACTION5).
- Clicking the grey shape also selects it, without using a counter cell; clicks on the yellow shape, bar, target or background do nothing (L2_E04).
- With the grey shape selected, the controls work like level 1: ACTION1 = both shapes up 3, ACTION3 = grey left / yellow right (L2_E03).
- Interpretation consistent with all of the above: the yellow shape is the reflection of the grey shape in the bar (mirror).
- The yellow shape can leave the screen on the left (L2_E02_ACTION3_x5, step 5: yellow no longer visible).
- LEVEL 2 SOLVED in 11 actions: 2 x ACTION3 (bar left, yellow 12 cells left), select the grey shape (ACTION5 or click), 8 x ACTION2 (L2_E07, L2_E07b).

## Open questions
- Was the grey shape the selected object in level 1 too (it carried the dots)? Could ACTION5 have selected the bar there?
- What happens when the counter column is full?
- What new mechanism does level 3 introduce?
