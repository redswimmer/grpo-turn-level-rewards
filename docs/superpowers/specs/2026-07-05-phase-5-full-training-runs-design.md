# Phase 5 Full Training Runs — Design

Status: approved by user, 2026-07-05. Implements `docs/phase-5-full-training-runs.md`.

## Context and evidence gathered

- Confirmed via `git log --all` / `gh pr list`: Phases 1-4 are merged to `main` (PR #4 landed
  2026-07-05). This work happens on a new `phase-5-full-training-runs` branch, not `main`.
- Confirmed via `grep` on the actual source (not assumed from the phase doc's prose): neither of
  Phase 5's two planned code changes exist yet in `train.py`/`rewards.py` — the phase doc's "Not
  started" status is accurate.
- **Real discrepancy found and resolved**: the phase doc's "Config" section says `--train-size`
  should be left unset to get "all rows" via `n=None`, but `train.py`'s actual `_parse_args`
  defaults `--train-size` to `8` (an `int` argparse field, no `None` sentinel). The phase doc
  itself already anticipated this exact gap ("pass `--train-size 90447` explicitly if `train.py`'s
  own CLI default differs... check `_parse_args` first rather than assuming") — resolved by
  passing `--train-size 90447` explicitly in the launch commands below. `load_train_dataset(n=90447)`
  after shuffling is equivalent to `n=None` since the filtered dataset has exactly 90,447 rows; no
  CLI code change needed.
- Retrieval server confirmed running (`localhost:8000`, PID checked live) and GPU confirmed idle
  (173MiB/24564MiB used) before this design was finalized.
- Phase 4's smoke test only validated `num_generations=2`; Phase 5 jumps to the paper's
  `num_generations=21` (10.5x larger, since `per_device_train_batch_size` is forced equal to
  `num_generations`). Whether `gradient_checkpointing=True` alone (Phase 4's OOM fix at the smaller
  scale) survives this jump is a genuine unknown — not resolved by reading code, must be measured.

## Sequence

1. **Canary dry-run** (this session, foreground) — de-risk `num_generations=21` before writing any
   code.
2. **Code changes** to `train.py` / `rewards.py` (below), gated on the canary succeeding or its
   OOM fallback resolving.
3. **Static verification gate** (`scripts/verify_phase5.py`) — must pass before any full run.
4. **Launch `outcome_only` full run** in background; verify healthy for ~20-30 steps; hand off for
   user monitoring.
5. **Launch `turn_level` full run** once the user confirms `outcome_only` is healthy.
6. **Post-run evidence gate** for each condition once it finishes (below) — before Phase 5 is
   considered done.
7. **README updates** — restructure and populate with results (below). Independent of the
   training-run mechanics but part of this phase's scope.

## Canary dry-run

```
python -m turn_level_rewards.train --condition outcome_only --num-generations 21 --max-steps 3
```
`--train-size`/`--eval-size` stay at their CLI default (`8`) — the canary only needs the real
batch shape, not real data volume. Watched directly: `nvidia-smi` peak memory, per-step
wall-clock.

Decision gate:
- **Clean** → proceed to code changes with `gradient_checkpointing=True` confirmed sufficient at
  this scale.
- **OOM** → attempt installing `flash-linear-attention`/`causal-conv1d` first (per Phase 4's notes,
  this is the "real" fix for Qwen3.5's hybrid linear-attention layers; risk: uncertain to build
  cleanly against `torch==2.12.1+cu130`). Report back before spending significant time on the
  compile if it looks unlikely to succeed, rather than silently falling back to reducing
  `num_generations` (a paper-fidelity deviation that needs its own sign-off, not a silent default).

## Code changes

**`train.py`'s `build_config`** — new fixed fields, identical across both conditions:
- `num_iterations=2` (paper's Appendix E.3: "two training iterations per batch")
- `eval_strategy="steps"`, `eval_steps=20` (~15 eval checkpoints over 300 steps) — GRPO's analog to
  an eval-loss curve, using the already-wired `eval_dataset`
- `save_strategy="steps"`, `save_steps=50`, `save_total_limit=3` — crash-resilience checkpoints
  every 50 steps, capped at 3 retained to bound disk usage (300 is divisible by 50, so the final
  checkpoint always lands naturally)

New `tests/unit/test_train.py` case: assert all of the above on `build_config`'s return value for
both conditions.

**`rewards.py`** — `log_metric` calls so trackio gets clean per-metric curves instead of only
composite reward values:
- `outcome_reward`: log `exact_match`, `f1` per-completion
- `format_reward`: log `format_compliance_rate` (0/1 per-completion)
- `turn_reward`: log `retrieval_fraction` (unscaled, before the 0.4 magnitude weight)

New `tests/unit/test_rewards.py` case: fake `log_metric` callable (this repo's existing DI seam
pattern — inject the seam, assert what was logged), not a real trackio backend.

## Full run launches

```
python -m turn_level_rewards.train --condition outcome_only --max-steps 300 --num-generations 21 --eval-size 64 --train-size 90447
python -m turn_level_rewards.train --condition turn_level   --max-steps 300 --num-generations 21 --eval-size 64 --train-size 90447
```
Sequential (single GPU), both backgrounded. `outcome_only` launches first as the simpler-reward
canary (per CLAUDE.md's established verification ordering); `turn_level` launches only after the
user confirms `outcome_only` looks healthy.

## Verification plan

Two layers — replacing the earlier draft's soft "no unresolved alerts / curves visible" language
with concrete, evidence-based checks:

**A. Static gate — `scripts/verify_phase5.py`**, mirrors the existing `verify_phase{2,3,4}.py`
pattern, run immediately after the code changes and before any GPU time is spent:
- `pytest tests/unit/`, `ruff check`, `ty check`
- Assert `build_config`'s new fields (`num_iterations`, `eval_strategy`/`eval_steps`,
  `save_strategy`/`save_steps`/`save_total_limit`) for both conditions

**B. Post-run evidence gate**, run after each condition's 300 steps finish:
1. **Checkpoint loads**: `AutoModelForCausalLM.from_pretrained("outputs/{condition}")` actually
   succeeds — not just "the directory exists."
2. **Alerts**: `trackio list alerts --project turn-level-rewards --json --since <run-start-ts>` —
   empty, or every entry individually explained in this doc's Handoff notes.
3. **Step count**: pull the run's logged step metric, confirm it reached 300 (catches silent early
   termination).
4. **Curve completeness**: `trackio get metric ... --json` for each of the 4 new per-metric curves
   plus the periodic eval-reward curve — confirm a real, non-trivial number of non-null points
   (~300 for per-step metrics, ~15 for eval), not just that the metric name exists.
5. **Judgment read** (not scriptable): does eval-reward plateau, diverge from, or track training
   reward — same as Phase 4's manual transcript reading, human/agent judgment stays here.

Phase 5 is only considered done once both layers pass for both conditions.

## README updates

Two changes, independent of the training-run mechanics above:

1. **Restructure** `README.md` so "what we found" and "how to run it" are separately scannable,
   ordered for the reader who wants findings first:
   - `## What this compares` (unchanged)
   - `## Results` (new) — self-contained: written findings and key numbers live directly in the
     text, comparing each result's direction to the paper's own reported finding. No pointer to
     "run `trackio show` to see for yourself" — a reader should never need to execute anything to
     understand what was found.
   - `## Reproducing this` (renamed from today's Quick Start/Retrieval/Training sections) — purely
     mechanical (prerequisites, retrieval server setup, training commands), moved after Results,
     for the reader who wants to get in the weeds.
   - `## Contributing` (unchanged, stays last)
   - Also drop the existing `docs/phase-5-full-training-runs.md` reference in the current Training
     section — no phase references belong in the README.
2. **Populate `## Results`** once each full run's post-run evidence gate passes: a concise
   per-condition entry (a sentence or two, or a small table) stating the key metric (e.g. final
   `exact_match`/`f1`, `retrieval_fraction`) and how it compares to the paper's own reported
   direction of effect — not a data dump, not editorializing. This is the first entry in what
   should become a standing practice: every future phase that produces experimental results
   updates this same section the same way.

## Exit criteria

- [ ] Canary dry-run completed (clean, or OOM fallback resolved) and reported.
- [ ] `scripts/verify_phase5.py` passes.
- [ ] Both full runs complete all 300 steps.
- [ ] Post-run evidence gate (layer B above) passes for both conditions.
- [ ] Both checkpoints saved and confirmed loadable.
- [ ] README restructured (`Results` / `Reproducing this` split) and `Results` populated for both
      conditions.

## Handoff notes

(not yet started)
