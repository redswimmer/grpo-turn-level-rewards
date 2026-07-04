# Phase 5: Full training runs

## Goal

Run both reward conditions to completion at the scale recommended in CLAUDE.md, producing two
trained checkpoints and full trackio training curves, ready for comparison in Phase 6.

## Read first

`CLAUDE.md` — "Reward design", "Hardware", "Experiment tracking (trackio)". Also read Phase 4's
Handoff notes for the real observed per-step wall-clock time — use that to sanity-check
`--max-steps` before launching a long run rather than assuming CLAUDE.md's original estimate
still holds.

## Prerequisites (entry state)

- Phase 4's smoke test passed for both conditions.
- Retrieval server running and stable (it needs to stay up for the full duration of both runs).

## Tasks

- [ ] Launch the `outcome_only` full run in the background (`train.py --condition outcome_only`
      with the full recommended `--train-size`/`--max-steps`). Watch the first ~20-30 steps via
      the trackio dashboard/alerts before considering it healthy and moving on (per CLAUDE.md's
      verification-approach ordering: simpler reward first, as a canary).
- [ ] Once `outcome_only` looks healthy, launch the `turn_level` full run the same way.
- [ ] Let both runs complete; do not babysit continuously — poll via
      `trackio list alerts --project turn-level-rewards --json --since <timestamp>` rather than
      watching stdout the whole time.
- [ ] Save both final checkpoints to `outputs/{condition}/`.

## Exit criteria (all must be true before handing off)

- [ ] Both runs completed all planned steps without crashing.
- [ ] No unresolved trackio alerts from either run (any that fired were investigated and are
      understood — e.g. an expected early-training dip vs. a genuine problem).
- [ ] Both checkpoints saved and loadable.
- [ ] Full reward/metric curves for both runs are visible in the shared trackio project.

## Handoff notes

<!-- Fill in after completing this phase: actual step counts / wall-clock time for each run, any
alerts that fired and what they meant, checkpoint paths, and anything unusual observed in the
curves that Phase 6 should investigate specifically (e.g. did turn_level's retrieval_fraction
plateau near the ~80% corpus ceiling documented in CLAUDE.md, or somewhere else?). Leave this
section for the next fresh agent to read first. -->

(not yet started)
