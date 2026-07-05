# Phase 5: Full training runs

## Goal

Run both reward conditions to completion at a scale anchored to the source paper's own GRPO
ablation (Appendix E.3), producing two trained checkpoints and full trackio training curves
(including periodic held-out reward, GRPO's analog to an "eval loss" curve), ready for comparison
in Phase 6.

## Read first

`CLAUDE.md` — "Reward design", "Hardware", "Experiment tracking (trackio)", "TRL mechanics being
relied on" (specifically the `num_iterations` = μ reference). Also read Phase 4's Handoff notes
(`docs/phase-4-training-smoke-test.md`) for the real observed per-step wall-clock time and the
three real bugs + one CUDA OOM the smoke test caught — this phase runs at a much larger
`num_generations` (21 vs. the smoke test's 2), so re-read the OOM section before assuming
`gradient_checkpointing=True` alone is sufficient at this scale; if it isn't, the next thing to
try is installing `flash-linear-attention`/`causal-conv1d` properly (not attempted in Phase 4).

## Prerequisites (entry state)

- Phase 4's smoke test passed for both conditions (done — see `docs/phase-4-training-smoke-test.md`).
- Retrieval server running and stable (it needs to stay up for the full duration of both runs;
  relaunch command is in `docs/phase-1-retrieval-infra.md`'s Handoff notes if it's died).

## Config, grounded in the paper (not guessed)

Fetched directly from arXiv:2505.11821's Appendix E.3 ("Experiment Setup") for its GRPO-OR/GRPO-MR
case study:
- Total training steps: **300**.
- Rollout generations per prompt (group size): **21**.
- "Each batch undergoes two training iterations" — this is TRL's `num_iterations` field (already
  referenced in CLAUDE.md's TRL mechanics section as "= μ in the GRPO paper", but never set in
  `build_config` since Phase 4 only needed the default of 1).
- The paper's own per-device-batch/gradient-accumulation split (12 × 4 = 48) is **not**
  reproducible under TRL's `generation_batch_size % num_generations == 0` constraint (48 isn't
  divisible by 21) — the paper likely isn't using TRL internally. This repo keeps its own
  already-established pattern instead: `per_device_train_batch_size = num_generations = 21`
  (trivially divisible), not an attempt to replicate the paper's exact batch/grad-accum split.
- Dataset size: the paper doesn't state a literal unique-question count (step-based training with
  resampling). At `generation_batch_size=21` and `num_generations=21`, each step samples
  **exactly one unique prompt** (repeated 21 times for the rollout group) — so `--train-size`
  doesn't need precise sizing; use `--train-size` unset via `n=None` (the full 90,447-row
  dataset, shuffled) so there's no risk of exhausting/over-repeating rows across 300 steps.

**Concrete launch config for both conditions** (only `--condition` differs, per CLAUDE.md's own
stated invariant):
```
--max-steps 300 --num-generations 21 --eval-size 64
```
(`--train-size` left at its data.py default of "all rows" — pass `--train-size 90447` explicitly
if `train.py`'s own CLI default differs from that by the time this phase runs; check `_parse_args`
first rather than assuming.)

## Tasks

- [ ] **Add `num_iterations=2` and periodic in-training eval to `build_config`** (extends
      `src/turn_level_rewards/train.py`, already-existing function from Phase 4): add
      `num_iterations=2` (matches the paper's "two training iterations per batch") and
      `eval_strategy="steps"`, `eval_steps=20` (roughly 15 eval checkpoints across a 300-step run)
      as new fixed fields in the returned `GRPOConfig` — identical across both conditions, same as
      every other fixed field Phase 4 already established. This is the mechanism for GRPO's real
      analog to an "eval loss" curve: GRPO's own training loss doesn't indicate fit quality the
      way SFT's cross-entropy loss does (a near-zero loss can just mean a step's group had
      identical rewards, not that the policy converged) — periodic reward computed on the
      **held-out** `eval_dataset` (already wired into `build_trainer` since Phase 4, currently
      unused because `eval_strategy` defaults to `"no"`) is the actual signal for
      plateau/overfitting, matching the paper's own Table 5 "validation reward" tracking. Add a
      `tests/unit/test_train.py` case asserting both new fields on the built config. Use
      `--eval-size 64` (a small, fast periodic subset) — distinct from Phase 6's `evaluate.py`,
      which runs the *full* 7,405-row held-out set once, post-hoc, on the final checkpoint; the
      two don't need to match in size.
- [ ] **Add `log_metric` calls to `rewards.py`'s reward functions** (extends
      `src/turn_level_rewards/rewards.py`, already-merged Phase 2 code): CLAUDE.md's "Experiment
      tracking" section always intended `retrieval_fraction`, `exact_match`, `f1`, and
      `format_compliance_rate` to land in trackio as clean, separate per-step curves via TRL's
      `log_metric` reward-function kwarg (confirmed real and already wired by TRL —
      `grpo_trainer.py:1330-1359` calls `reward_kwargs["log_metric"] = self._log_metric` — not a
      hypothetical feature). This was never actually implemented in Phase 2 or Phase 4: right now
      trackio only sees the *composite* reward values (`rewards/outcome_reward/mean` = F1 + 0.5·EM
      entangled together, `rewards/turn_reward/mean` = 0.4×retrieval_fraction scaled), with no way
      to see e.g. "did raw EM go up" or "did retrieval rate actually improve" as its own curve.
      Add `log_metric` calls inside `outcome_reward` (log `exact_match`, `f1` per-completion),
      `format_reward` (log `format_compliance_rate` as a 0/1 per-completion value), and
      `turn_reward` (log `retrieval_fraction` directly, unscaled by the 0.4 magnitude weight) —
      `log_metric` averages across the batch automatically per TRL's own doc comment. Cover with a
      `tests/unit/test_rewards.py` case using a fake `log_metric` callable (matching this repo's
      existing DI pattern — inject the seam, assert what was logged), not a real trackio backend.
- [ ] Launch the `outcome_only` full run in the background (`train.py --condition outcome_only
      --max-steps 300 --num-generations 21 --eval-size 64`). Watch the first ~20-30 steps via the
      trackio dashboard/alerts before considering it healthy and moving on (per CLAUDE.md's
      verification-approach ordering: simpler reward first, as a canary).
- [ ] Once `outcome_only` looks healthy, launch the `turn_level` full run the same way
      (`--condition turn_level`, identical other flags).
- [ ] Let both runs complete; do not babysit continuously — poll via
      `trackio list alerts --project turn-level-rewards --json --since <timestamp>` rather than
      watching stdout the whole time. Also watch the new periodic `eval_reward` curve (from the
      task above) for a plateau or divergence from the training reward curve — that's the
      overfitting/convergence signal this phase specifically adds.
- [ ] Save both final checkpoints to `outputs/{condition}/` (already the default `output_dir` per
      `build_config` — confirm the checkpoint actually lands there and is loadable, don't just
      assume the default path was honored).

## Exit criteria (all must be true before handing off)

- [ ] Both runs completed all planned steps (300) without crashing.
- [ ] No unresolved trackio alerts from either run (any that fired were investigated and are
      understood — e.g. an expected early-training dip vs. a genuine problem).
- [ ] Both checkpoints saved and loadable.
- [ ] Full reward/metric curves for both runs are visible in the shared trackio project, including
      the new periodic eval-reward curve and the new per-metric (`exact_match`/`f1`/
      `retrieval_fraction`/`format_compliance_rate`) curves from this phase's two code tasks.

## Handoff notes

<!-- Fill in after completing this phase: actual step counts / wall-clock time for each run
(compare against Phase 4's ~7-9s/step-at-num_generations=2 figure to see how generation cost
scaled with group size), any alerts that fired and what they meant, whether gradient_checkpointing
alone was sufficient to avoid OOM at num_generations=21 or the fused kernels needed installing
after all, checkpoint paths, the periodic eval-reward curve's shape (did it plateau, diverge from
train reward, or track it closely?), and anything unusual observed in the curves that Phase 6
should investigate specifically (e.g. did turn_level's retrieval_fraction plateau near the ~80%
corpus ceiling documented in CLAUDE.md, or somewhere else?). Leave this section for the next fresh
agent to read first. -->

(not yet started)
