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

- [x] **Add `num_iterations=2` and `save_strategy`/`save_steps`/`save_total_limit` to
      `build_config`** — done, but **periodic in-training eval was deliberately dropped**, a real
      deviation from this doc's original plan: `GRPOTrainer`'s `environments` pool (required by
      `SearchEnv`) is built once at init, sized to train's `generation_batch_size`, and reused
      unconditionally for both train and eval in the installed `trl==1.7.1` — a differently-sized
      eval batch raises a `ValueError`, and matching eval's batch to train's reproduces the exact
      OOM this phase's micro-batching fix exists to solve, with no eval-side chunking knob
      available. This is already fixed upstream (TRL PR #6001, commit `8b61980d`) but not in any
      released version — see CLAUDE.md's "TRL mechanics" section for the full writeup and the
      revisit condition. Phase 6's `evaluate.py` is the only held-out signal now.
- [x] **Add `log_metric` calls to `rewards.py`'s reward functions** — done as originally planned;
      `exact_match`/`f1`/`format_compliance_rate`/`retrieval_fraction` all land in trackio as clean
      per-step curves.
- [x] Launch the `outcome_only` full run, verify healthy, launch `turn_level` — done, but **not on
      the first or second attempt**. See Handoff notes below for two real bugs the live runs
      surfaced (an OOM that survived the original micro-batching fix, and a silent policy collapse
      from a follow-on batching bug) plus an unrelated infrastructure issue (a transient systemd
      cgroup killing the training process independent of anything training-related).
- [x] Let both runs complete, checkpoints saved to `outputs/{condition}/` and confirmed loadable.

## Exit criteria (all must be true before handing off)

- [x] Both runs completed all planned steps (300) without crashing.
- [x] No unresolved trackio alerts from either run.
- [x] Both checkpoints saved and loadable (`outputs/outcome_only/checkpoint-300`,
      `outputs/turn_level/checkpoint-300`; confirmed via `AutoModelForCausalLM.from_pretrained`).
- [x] Full reward/metric curves for both runs are visible in trackio (150 points each for
      `reward`/`exact_match`/`f1`/`format_compliance_rate`, plus `retrieval_fraction` for
      `turn_level`). **No periodic eval-reward curve** — dropped, see Tasks above.

## Handoff notes

**Config as actually run** (differs from this doc's original plan in two ways, both load-bearing):
- `--max-steps 300 --num-generations 21 --eval-size 64 --train-size 90447` for both conditions —
  the launch flags matched the plan, but `build_config` internally sets
  `per_device_train_batch_size=1` (not `21`) and `gradient_accumulation_steps=21`, not the
  `per_device_train_batch_size=num_generations=21` this doc originally assumed. See the two bugs
  below for why.
- **150 distinct training prompts, not 300**: `num_iterations=2` means each sampled prompt's
  rollout group is reused for 2 real optimizer steps (confirmed directly: trackio's logged
  `train/global_step` for reward-bearing entries advances by exactly 2 per prompt — 1, 3, 5, 7,
  ...). 300 total steps ÷ 2 = 150 distinct prompts. **This doc's original claim that "each step
  samples exactly one unique prompt" was imprecise** — it didn't account for `num_iterations`'
  reuse. Whether this matches the paper is genuinely unresolved: refetched arXiv:2505.11821
  Appendix E.3 directly, and while it states both "total training steps is set to 300" and "each
  batch undergoes two training iterations," **it never clarifies whether the 300 already includes
  the 2-iteration reuse or not** — the two sentences appear with no stated relationship. This
  repo's 150-distinct-prompts/300-total-steps reading follows the standard ML convention (a "step"
  = one gradient update, matching TRL's own `global_step`), which is principled but not a verified
  1:1 match — a genuine interpretive call given real ambiguity in the source text, not a slip.

**Bug #1 (OOM) — real canary run, `num_generations=21`, `per_device_train_batch_size=21` (no
chunking)**: TRL's `_get_per_token_logps_and_entropies` tried to allocate 28.29 GiB for one
logits-to-fp32 conversion, OOMing the 24GB GPU. A chunk size of 3 (`per_device_train_batch_size=3`)
still OOMed on the backward pass with a smaller shortfall (confirmed not a fragmentation artifact —
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` made no difference). Fixed by capping
`per_device_train_batch_size` at 1 (fully sequential, single-sequence chunks) — see
`_MAX_TRAIN_MICRO_BATCH_SIZE`/`_train_micro_batch_size` in `train.py`.

**Bug #2 (silent policy collapse) — a real 6300-step run** (launched at that scale before Bug #1's
fix above was itself buggy) **collapsed into a fixed, zero-variance `-0.1` reward
(`frac_reward_zero_std=1.0`) for 20+ consecutive rollout groups**, starting right after the 8th
group. Root cause, confirmed via direct sqlite query of the run's own metrics and the logged
completions transcript (garbled, malformed tool-call syntax, identical across all 21 samples):
the OOM fix made up the difference from capping `per_device_train_batch_size` via
`steps_per_generation=21` directly, leaving `gradient_accumulation_steps` at its default of `1`.
TRL's own docstring says `steps_per_generation` "defaults to `gradient_accumulation_steps`" when
unset — the intended lever is `gradient_accumulation_steps`. With it left at 1, each of the 21 (×2
for `num_iterations`) memory-safe micro-batches triggered its own independent optimizer step
instead of being combined into one properly-averaged update per rollout group — noisy
single-sequence updates instead of stable group-averaged ones. Fixed by setting
`gradient_accumulation_steps` instead and leaving `steps_per_generation` unset so it defaults to
match. Verified via a fresh 500-step run showing real reward variance well past the point where the
old run had collapsed, before committing to the real full runs.

**Infrastructure issue (unrelated to training code)**: two consecutive `outcome_only` launches were
killed externally at nearly identical elapsed time (~25-32 min), regardless of training progress —
no CUDA OOM, no traceback, system RAM/disk fine. Traced to the interactive session running inside
`app-ghostty-surface-transient-*.scope`, a systemd scope tied to the terminal window's lifecycle,
which was getting reaped and killing everything in its cgroup. Fixed by launching training via
`systemd-run --user --scope --unit=<name> -- <command>`, placing it in an independent,
non-transient scope. Confirmed via `cat /proc/<pid>/cgroup` and by the process surviving well past
the previous kill point. Relevant for any future long (>~25 min) background process in this
environment, not specific to this repo's code.

**Actual wall-clock**: `outcome_only` — 300/300 steps in 1h26m46s (~17.4s/step average).
`turn_level` — 300/300 steps in 1h40m16s (~20.1s/step average, slower likely due to the extra
`turn_reward`/retrieval-server round trip per rollout). Both notably slower than Phase 4's
2-generation smoke test (~6-9s/step) but far faster than the (incorrect) ~4.5-5 hour/condition
estimate computed before Bug #2's fix was found — that estimate was itself based on the same wrong
steps-per-generation mental model as Bug #2.

**Results** (training-batch metrics, first-third vs. last-third of each run's 150 logged points —
not a held-out evaluation; that's Phase 6's job):

| Metric | outcome_only (first → last third) | turn_level (first → last third) |
|---|---|---|
| exact_match | 0.170 → 0.224 | 0.213 → 0.233 |
| F1 | 0.245 → 0.311 | 0.296 → 0.340 |
| format_compliance_rate | 0.949 → 0.996 | 0.882 → 0.990 |
| retrieval_fraction | n/a | 0.414 → 0.314 |

Both conditions show real learning (EM/F1 trending up). `turn_level` starts and ends with somewhat
higher EM/F1 than `outcome_only`, directionally consistent with the paper's hypothesis, but with
only 150 training prompts each this is a small-scale training-batch signal, not a rigorous claim —
Phase 6's full 7,405-row held-out evaluation is the real test. **Flag for Phase 6**:
`turn_level`'s `retrieval_fraction` trended *down* (0.41 → 0.31) rather than up over training —
worth checking directly in the held-out evaluation rather than assuming it's noise.

**No periodic eval-reward curve** (dropped — see Tasks above), so no plateau/divergence signal to
report here; Phase 6's held-out `evaluate.py` run is the first real generalization check.
