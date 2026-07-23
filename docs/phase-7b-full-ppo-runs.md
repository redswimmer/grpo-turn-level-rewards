# Phase 7b: Full PPO/MT-PPO training runs + evaluation + comparison

## Goal

Run both `ppo` and `mt_ppo` (deterministic rewards) to completion, evaluate both checkpoints on
the full held-out set, and produce the actual comparison — the paper's real Table 2 headline
result (`PPO-OR` vs `MT-PPO`), reproduced on this repo's HotpotQA/wiki-18 pipeline. Also replace
the README's current plain charts with a matplotlib-driven visual system designed to actually
tell the comparison's story, not just plot numbers.

This phase exists because Phase 7's own doc and Phase 8's own doc both stop at smoke-test scale
— see CLAUDE.md's Roadmap section (2026-07-23 addendum) for why that left a real gap. 7b is the
PPO-track analog of Phases 5+6 combined (full runs, then `evaluate.py`/`compare_runs.py`/write-up)
for the GRPO track.

## Read first

`CLAUDE.md`'s Goal section (recall: PPO/MT-PPO's Table 2 numbers are the paper's actual
best-benchmarked result, not a secondary ablation). Phase 7's Handoff notes
(`docs/phase-7-mt-ppo.md`) — checkpoint paths, real per-step wall-clock observed, any TRL/critic
surprises, and whatever logging mechanism Phase 7 actually wired up (this doc doesn't assume
trackio integration works identically to `GRPOTrainer`'s automatic one, since `MTPPOTrainer` is
built directly on `transformers.Trainer` — confirm what Phase 7 actually did before assuming).
Phase 6's doc (`docs/phase-6-evaluation-comparison.md`) for the shape of a real evaluation +
comparison write-up, including its "is more training needed?" validation checklist and its
follow-up-experiments pattern — reuse that structure rather than reinventing it.

## Prerequisites (entry state)

- Phase 7 done: `MTPPOTrainer` works for both `ppo`/`mt_ppo` conditions, smoke-tested, per
  `docs/phase-7-mt-ppo.md`'s exit criteria.
- Retrieval server running and stable.

## Tasks

- [ ] Full training runs: both `ppo` and `mt_ppo`, 500 rollout-collection steps × 4 inner PPO
      epochs per the paper's Section 6.2/C.1.3 spec (already recorded in
      `docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md`). Confirm real wall-clock
      before committing to the full budget — Phase 7's smoke test only ran a couple of steps;
      PPO's extra critic forward/backward pass and multiple inner epochs make this a different
      cost profile than GRPO's Phase 5 runs, not an assumed-similar one.
- [ ] A held-out evaluation path for `MTPPOTrainer` checkpoints. **This is not just "reuse
      `evaluate.py`"** — Phase 6's `evaluate.py` works by constructing a `GRPOTrainer` and calling
      its standard `.evaluate()`, which relies on `GRPOTrainer`'s own `prediction_step` override.
      `MTPPOTrainer` is a different class with its own rollout loop; the eval path needs to either
      (a) reuse `MTPPOTrainer`'s own rollout loop directly (no gradient updates, same
      `SearchEnv`/`rewards.py` machinery) against the held-out set, or (b) some other mechanism —
      resolve this as a real design question when this phase starts, don't assume (a) is right
      without checking whether the rollout loop cleanly separates from the update step.
- [ ] Comparison write-up: EM/F1/retrieval-fraction, `ppo` vs `mt_ppo`, mirroring Phase 6's
      "is more training needed?" validation checklist before treating any result as final.
- [ ] Matplotlib visuals (new — not in the original Phase 7 design): the current README charts are
      functional but plain; redesign them to actually communicate the comparison's story clearly.
      **Use the `dataviz` skill** when doing this chart work, not ad hoc matplotlib defaults.
      Decide during this phase (not assumed here) whether to also retrofit the existing
      `outcome_only`/`turn_level` charts to the same new visual system for consistency, or leave
      those as-is and only apply the new system to the `ppo`/`mt_ppo` results.
- [ ] README: add a Results section for `ppo`/`mt_ppo`, following the existing section's
      reader-focused structure (self-contained, grounded in the paper's own numbers, no internal
      doc citations) but using the new chart system.

## Exit criteria (all must be true before handing off)

- [ ] Both `ppo` and `mt_ppo` checkpoints exist, are loadable, and completed their full training
      budget.
- [ ] Held-out evaluation completed for both, using a real (not placeholder) eval path.
- [ ] A comparison verdict recorded (real advantage, no meaningful difference, or inconclusive —
      state it plainly either way, same standard Phase 6 held itself to).
- [ ] New matplotlib charts committed and embedded in the README, reviewed for actually being
      clearer than the prior charts (not just different).

## Handoff notes

<!-- Fill in after completing this phase: real wall-clock/cost for the full runs, the eval-path
design actually chosen and why, the comparison verdict and numbers, and anything about the new
chart system worth carrying into Phase 8b (e.g. a reusable plotting module, a settled color/style
convention). Leave this section for the next fresh agent to read first. -->

(not yet started)
