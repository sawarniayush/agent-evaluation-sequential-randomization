# Analysis data

Each study directory contains `panel.npz`, `metadata.json`, `parameters.json`, and `estimates.csv`. All arrays are numerical and load with `numpy.load(..., allow_pickle=False)`.

| Array | Meaning |
| --- | --- |
| `task_index`, `episode_id` | Zero-based complete-trajectory index for each decision row |
| `t` | Decision position; ImpossibleBench retains chronological checkpoint numbers 1–5 |
| `treatment` | Observed assignment, 0 or 1 |
| `reward` | Terminal outcome repeated on each decision row |
| `outcome_sums` | One terminal outcome for every complete trajectory, including those with no decisions |
| `fold` | Held-out fold for each decision; whole trajectories share a fold |
| `x` | Ordered history columns for outcome regressions |
| `target_control_x` | Ordered current-history inputs for future-feature regressions |
| `f` | Raw effect features before phase expansion or SVD |
| `stage` | Monotone progress phase or chronological checkpoint |
| `group_masks` | Boolean membership matrix; column order matches `metadata.json` |
| `m0`, `m1`, `q` | Saved out-of-fold outcome predictions |

`metadata.json` names every effect, outcome-history, and future-regression column and defines the groups. `parameters.json` records the fitted model settings. `estimates.csv` contains the paper estimates on a 0–100 scale, confidence limits, standard errors, assignment counts, and support flags. An empty numerical cell denotes an unavailable estimate or interval.

Retail contains 17,807 decision rows for GPT-5 nano and 11,247 for Claude Haiku. ToolSandbox contains 10,932 and 8,981, respectively. ImpossibleBench contains 1,771 eligible checkpoints.

Retail and ImpossibleBench contrasts are percentage points. ToolSandbox contrasts are score points. Positive ImpossibleBench effects mean more detected rule violations. Group membership may overlap, so group counts must not be added to obtain a trajectory count.

The `retail_mc` directory contains policy identifiers, task identifiers, replicate indices, and terminal outcomes for all 4,560 direct-policy trajectories. Its table compares direct gains with the Retail SNMM estimates.

The five randomized-study directories also contain `plot_rows.csv`, which combines the same estimates with the paper's display labels. Original figure PDFs are in `figures/`; regenerated statistical figures go to the output directory specified on the command line.
