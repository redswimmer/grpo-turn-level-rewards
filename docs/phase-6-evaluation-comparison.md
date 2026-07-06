# Phase 6: Evaluation + comparison

## Goal

Evaluate both trained checkpoints on the full held-out set and produce the final comparison — the
actual answer to "does turn-level reward help, and how, in this simplified recreation."

## Read first

`CLAUDE.md` — "Goal" (recall the exact axes the paper itself compares on: accuracy,
stability/convergence speed, and the retrieval-rate mechanism believed to explain it), "TRL
mechanics being relied on" (the periodic-eval/`environment_factory` incompatibility note — relevant
context even though this phase's `evaluate.py` sidesteps it, see below), and "Experiment tracking
(trackio)". Read Phase 5's Handoff notes (`docs/phase-5-full-training-runs.md`) in full — checkpoint
paths, the two real training bugs, the genuine ambiguity in the paper's "300 steps" wording (this
repo's runs are 150 distinct training prompts per condition, not necessarily the paper's own
literal scale), and first training-batch results (both conditions learned; `turn_level`
directionally ahead of `outcome_only` on EM/F1 — this phase's job is to check whether that holds on
data neither model trained on).

## Prerequisites (entry state)

- Phase 5 done: both checkpoints exist and are loadable
  (`outputs/outcome_only/checkpoint-300`, `outputs/turn_level/checkpoint-300`); both runs' full
  training metrics are in trackio under run names `outcome_only-300steps-20260705-160524` and
  `turn_level-300steps-20260705-173317` (**not** the bare condition strings — `train.py` gives
  every invocation a unique, timestamped run name; check `trackio list runs --project
  turn-level-rewards --json` if these ever get superseded by a later run).

## `evaluate.py`'s design — resolved, not open (verified by real construction, not assumed)

The obvious-looking question — "how do you drive the multi-turn tool-calling generation loop
outside of `GRPOTrainer`, which owns all of that logic internally?" — has a clean answer:
**you don't drive it yourself; construct a `GRPOTrainer` for evaluation only and call its
public, standard `.evaluate()` method.** `GRPOTrainer` already overrides `prediction_step`
(`grpo_trainer.py:2862`, inherited from HF `Trainer`'s standard eval interface) to run the exact
same generation-and-score path used during training
(`_prepare_inputs` → `_generate_and_score_completions`). Confirmed by actually running it against
the real `outcome_only` checkpoint and the real retrieval server:

```python
config = GRPOConfig(
    output_dir=...,             # any writable scratch dir; nothing meaningful gets saved here
    num_generations=2,          # GRPO's hard minimum (>=2, needs group variance for advantages) --
                                 # we don't care about advantage quality at eval time, just the
                                 # per-completion reward/metric values, so the minimum is the
                                 # cheapest valid choice
    per_device_train_batch_size=2,  # must satisfy the same divisibility constraints as training,
    per_device_eval_batch_size=2,   # even though .train() is never called
    max_tool_calling_iterations=4,  # same as training
    beta=0.0,
    max_completion_length=2048,
    report_to="none",
)
trainer = GRPOTrainer(
    model="outputs/outcome_only/checkpoint-300",   # a local checkpoint path works transparently
    reward_funcs=get_reward_funcs(condition),
    args=config,
    train_dataset=data.load_train_dataset(n=2, seed=42),  # unused filler -- .train() never called
    eval_dataset=data.load_eval_dataset(n=None, seed=42),  # the full 7,405-row held-out set
    environment_factory=SearchEnv,
)
metrics = trainer.evaluate()
```

This returns a plain dict with everything `evaluate.py` needs, already computed by the exact same
reward functions training used (so eval and training reward are provably consistent, no
reimplementation): `eval_reward`, `eval_exact_match`, `eval_f1`, `eval_format_compliance_rate`, and
(for `turn_level` only, since `get_reward_funcs` includes `turn_reward`) `eval_retrieval_fraction`
and `eval_rewards/turn_reward/mean`. Confirmed real, sensible values from an actual run against 4
held-out rows and the trained `outcome_only` checkpoint: `eval_exact_match=0.25`, `eval_f1=0.45`,
`eval_format_compliance_rate=1.0`.

**Why this doesn't hit Phase 5's environment-pool bug**: that bug was two *different* batch sizes
(train's 21-generation group vs. a smaller eval batch) colliding within the *same* trainer
instance. Here, this trainer instance is used for evaluation only — one consistent batch shape for
its entire lifetime, so there's no mismatch to trigger it.

**Not yet verified — do this before committing to the full run**: the real full evaluation is
7,405 held-out rows, not 4. `per_device_eval_batch_size` can very likely go much higher than 2
without the OOM risk Phase 5 hit at training time, because unlike training (`num_generations=21`,
21 sequences all for the *same* prompt), eval only needs 2 sequences per prompt — so a larger
`per_device_eval_batch_size` (e.g. 32) packs multiple *different* held-out questions into one
generation call (16 questions/batch at `per_device_eval_batch_size=32`), which should meaningfully
cut wall-clock over one-question-at-a-time. Run a small canary at the target batch size and a
handful of rows first (mirroring Phase 5's own canary discipline) to pick a safe, fast batch size
and get a real per-row wall-clock estimate before launching the full 7,405-row pass — don't assume
either the batch size or the total wall-clock, measure them.

## Tasks

- [ ] `src/turn_level_rewards/evaluate.py` — for each checkpoint, build the eval-only `GRPOTrainer`
      above (`environment_factory=SearchEnv`, `eval_dataset=data.load_eval_dataset(n=None, ...)`
      for the full held-out set) and call `.evaluate()`; write the returned metrics dict to a json
      file (one per condition). Canary-verify the batch size first (see above).
- [ ] `scripts/compare_runs.py` — pull each condition's trackio metrics via `trackio get metric
      --project turn-level-rewards --run <run> --metric <name> --json`, using the **actual run
      names from Phase 5** above (not the bare condition strings) and the **`train/` metric-name
      prefix** (e.g. `train/reward`, `train/exact_match` — confirmed via direct sqlite query in
      Phase 5, not the bare names). Combine with both `evaluate.py` output jsons to produce: (1)
      overlaid EM/F1-vs-step curves for both conditions (the primary, directly-comparable
      convergence signal — **not raw composite reward**, since `turn_level`'s reward includes the
      extra `turn_reward` term and isn't on the same scale as `outcome_only`'s), (2) final held-out
      EM/F1 bar comparison, (3) final held-out retrieval-rate bar comparison, (4) `outcome_only`'s
      and `turn_level`'s `train/tools/call_frequency` curves side by side — see the specific paper
      claim below this is meant to check.
- [ ] Write up findings in plain language, scoped to the specific claim actually being tested:
      **the paper's own `GRPO-OR` vs `GRPO-MR` ablation.** The relevant numbers are in the paper's
      **Table 2** (re-verified directly, not assumed from memory — Table 2 reports `GRPO-OR`
      exact-match = 0, `GRPO-MR` = 0.3346; `MT-GRPO` = 0.5010 but that's the paper's separate
      turn-level-credit-assignment contribution, out of scope here per CLAUDE.md's Goal section).
      The paper also makes a specific, checkable *mechanism* claim, not just a final-number one:
      "`GRPO-OR` gradually stops calling search tools" during training, while the merged-reward
      condition keeps calling them — check this directly against `outcome_only`'s
      `train/tools/call_frequency` trend (already logged in trackio), not just final EM/F1. Does
      `turn_level` show higher held-out EM/F1 and a `retrieval_fraction`/tool-call-frequency
      trend that didn't decay the way `outcome_only`'s did? If the direction doesn't match the
      paper, is there a plausible reason already documented in CLAUDE.md or Phase 5's Handoff notes
      (the ~80% retrieval ceiling, the much smaller model/training scale, the genuine ambiguity in
      whether 150 distinct training prompts matches the paper's own scale)?

## Exit criteria (all must be true before handing off — this is the last phase)

- [ ] Both checkpoints evaluated on the same fixed, full 7,405-row held-out set.
- [ ] Comparison plots/tables produced and saved somewhere durable (not just a notebook cell).
- [ ] A short written summary of findings exists, explicitly stating whether this simplified
      recreation reproduces the paper's core claim or not, and why — including the specific
      tool-call-frequency-decay mechanism check above, not just final EM/F1.

## Handoff notes

<!-- This is the final phase — record final results and any recommended follow-up experiments
here (e.g. scaling up train size/steps, trying the closed-corpus or pooled-BM25 alternatives from
CLAUDE.md's rejected-options table as a fidelity check, tuning turn_reward's magnitude). -->

(not yet started)
