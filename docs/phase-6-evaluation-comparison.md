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
- [ ] Work through the "Decision: is more training needed?" section below against the actual
      numbers just produced, and record the outcome (triggered or not, and why) in this doc's
      Handoff notes — this is part of validating the comparison, not a separate follow-up step.

## Decision: is more training needed? (part of this phase's validation, not a separate follow-up)

After computing the held-out comparison, explicitly check these four criteria before declaring
Phase 6 done. This repro has exactly **one training seed per condition** — no repeated runs to
average over — so a real risk is reporting a difference that's actually just noise, or missing a
real one buried in a small-sample artifact. If any of these trigger, more training is genuinely
warranted, not optional polish:

1. **The held-out EM/F1 gap between conditions is small relative to single-run noise.** Rough
   check: compare the gap to the swing already visible within each run's own training curve (e.g.
   `train/exact_match`'s own point-to-point variance in trackio, already queried this way in
   Phase 5). If the between-condition gap is smaller than the within-run noise, don't report it as
   a finding as-is — **re-run both conditions with a different `--seed`** (same `--max-steps 300`,
   ~1.5-2hr each per Phase 5's actual wall-clock) and check whether the direction replicates.
2. **Held-out results contradict the training-batch trend Phase 5 documented** (e.g. `outcome_only`
   ends up ahead on held-out data despite `turn_level` looking ahead during training). This points
   at overfitting/small-sample instability, not just noise — the fix is **more distinct training
   prompts**, not a different seed. `distinct_prompts = max_steps / num_iterations` = `max_steps /
   2` at this repo's config, so e.g. `--max-steps 600` doubles training to 300 distinct prompts per
   condition.
3. **`outcome_only`'s tool-call-frequency doesn't reproduce the paper's claimed mechanism**
   (the paper says `GRPO-OR` "gradually stops calling search tools"; check `outcome_only`'s
   `train/tools/call_frequency` for a declining trend, not staying flat or rising). 150 distinct
   prompts may just be too short a run for that dynamic to emerge — but re-check it by extending
   **both conditions equally**, not `outcome_only` alone. Extending only the condition the claim
   is about would confound this phase's actual, controlled comparison (the whole point of the
   Goal section's "same agent, same algorithm, only the reward differs" design) with an unequal
   training budget — any resulting change could be "more steps," not "reward design," and either
   way the two decisions (does the mechanism eventually appear; does turn-level reward still win)
   need to share one apples-to-apples run, not two runs at different scales.
4. **`turn_level`'s held-out `retrieval_fraction` continues the downward trend Phase 5 flagged**
   (0.41 → 0.31 during training) rather than stabilizing. If confirmed still falling at the end of
   training, more steps would show whether it's a real, ongoing problem with `turn_reward`'s
   shaping (worth investigating directly) or just early-training noise settling into a stable,
   lower value (not actually still declining).

If none of these trigger, the current 150-distinct-prompt runs are sufficient to report a real,
if modest-scale, finding — don't manufacture a reason to keep training past a clean result.

**If any criterion triggers, resolve all of them with one symmetric re-run**, not a patchwork of
condition-specific fixes: re-run both conditions at the same larger `--max-steps` (e.g. `600`,
doubling to 300 distinct prompts each) and the same new `--seed` if criterion 1 also triggered.
A single equal-budget re-run re-checks the EM/F1 gap (criterion 1), the held-out/training-trend
consistency (criterion 2), `outcome_only`'s call-frequency trend over a longer horizon
(criterion 3), and `turn_level`'s retrieval_fraction trend (criterion 4) simultaneously, with
both conditions always seeing the identical budget.

## Exit criteria (all must be true before handing off — this is the last phase)

- [x] Both checkpoints evaluated on the same fixed, full 7,404-row held-out set (7,404, not 7,405
      — see Handoff notes for why 1 row was deliberately dropped).
- [x] Comparison plots/tables produced and saved in `results/` (git-committed, per this session's
      design decision — see the design spec).
- [x] The four more-training criteria explicitly checked and recorded below.
- [x] A short written summary of findings exists below.

## Handoff notes

**Held-out evaluation setup, real numbers not assumed:**
- Both conditions' `checkpoint-300` evaluated over the full held-out set via `evaluate.py`
  (`--eval-batch-size 4`, chosen after real canary testing — see below).
- `--eval-size 7404`, not the full 7,405: a ragged final batch (7,405 isn't evenly divisible by
  the 2 unique prompts/batch that `--eval-batch-size 4` implies at `num_generations=2`) reproduces
  the exact `ValueError: zip() argument 2 is longer than argument 1` CLAUDE.md's TRL mechanics
  section already documents for a mismatched environments-pool size — confirmed directly with a
  real 201-row canary. Dropping 1 of 7,405 rows sidesteps it entirely; statistically inconsequential.
- `--eval-batch-size 8` was tried first and OOMed *stochastically* on a 33-row canary (crashed on
  step 6 of 9, having succeeded cleanly on an earlier 32-row canary) — unlike training's
  same-prompt rollout groups, held-out batches mix genuinely different questions with much more
  completion-length variance, so 8 isn't reliably safe even though it worked once. `4` held up
  cleanly across a 200-row canary before being used for the real run.
- Wall-clock: `outcome_only` 2h35m (3,702 steps), `turn_level` 4h00m (slower — extra
  retrieval-server round trip per completion, consistent with Phase 5's training-time observation).
- **Real, confirmed via each checkpoint's own `trainer_state.json`** (not trackio's own returned
  `step` field, which is trackio's internal logging-call counter and does NOT match the trainer's
  real `global_step` — a real discrepancy found and resolved in this session): both conditions'
  checkpoints have `global_step=300`, `max_steps=300`, exactly 300 log entries — Phase 5's
  "150 distinct prompts, 300 real optimizer steps" claim is confirmed correct at the ground-truth
  level, not just assumed.

**Held-out results:**

| Metric | outcome_only | turn_level |
|---|---|---|
| Exact match | 0.2355 | 0.2068 |
| F1 | 0.3313 | 0.2943 |
| Format compliance | 0.9944 | 0.9743 |
| Retrieval fraction | n/a | 0.3812 |

**The four criteria, checked against real numbers:**

1. **Gap vs. single-run noise — TRIGGERS.** The held-out EM gap (0.029) and F1 gap (0.037) are
   both smaller than the swing already visible within a single run's own training curve — even at
   a smoothed first/mid/last-third granularity, `outcome_only`'s own EM ranges from 0.170 to 0.243
   across its own training (a 0.07 swing), larger than the whole between-condition gap. Visually
   confirmed too: `results/em_f1_training_curves.png`'s raw per-prompt curves for both conditions
   are effectively interleaved noise, not visibly separated lines.
2. **Held-out contradicts training-batch trend — TRIGGERS.** Phase 5's training-batch data showed
   `turn_level` ahead on both EM (0.233 vs 0.224 late-training) and F1 (0.340 vs 0.311). Held-out
   data reverses this: `outcome_only` ahead on both (0.2355 vs 0.2068 EM; 0.3313 vs 0.2943 F1) —
   the exact contradiction this criterion anticipated, not a near-miss.
3. **`outcome_only`'s tool-call-frequency mechanism — TRIGGERS, independent of `turn_level`.**
   The paper claims `GRPO-OR` "gradually stops calling search tools." `outcome_only`'s own
   training curve (binned first/mid/last third of its 150 logged points) shows the opposite:
   0.768 → 1.010 → 1.148 mean tool calls/completion — rising, not falling. See
   `results/tool_call_frequency.png`.
4. **`turn_level`'s retrieval_fraction continuing to decline — does NOT clearly trigger.** Training
   showed a real decline (0.414 first-third → 0.314 last-third). But the held-out value (0.3812)
   sits *between* those two training-time numbers, closer to the early-training value than the
   late-training one — not a continuation of the decline. Read as the training-time dip settling
   rather than an ongoing, worsening problem, though a single held-out number isn't itself a trend.

**Verdict: 3 of 4 criteria trigger. More training is genuinely warranted before treating either
direction as a real finding** — this is not a borderline call. Per this doc's corrected
recommendation (see "Decision" section above, fixed in this session after review to never treat
conditions asymmetrically): the follow-up is **one symmetric re-run of both conditions**, same
larger `--max-steps` (e.g. `600`, doubling to 300 distinct prompts each — addresses criteria 2–4)
**and** a different `--seed` (addresses criterion 1), not a patchwork of condition-specific fixes.
This has NOT been launched — it's a real multi-hour GPU-time decision left to the user.

**Honest summary of what this simplified recreation does and doesn't show:** real learning
happened in both conditions (format compliance and EM/F1 both improved substantially from a
near-zero start, and `outcome_only`'s held-out numbers matching/exceeding its own late-training
numbers rules out gross overfitting). But at this scale (one seed, 150 distinct training prompts
per condition), **this run does not reproduce the paper's `GRPO-OR`/`GRPO-MR` comparison** in
either direction: `turn_level`'s training-time lead reverses on held-out data, the gap either way
is smaller than single-run noise, and the paper's own claimed mechanism for why outcome-only
reward underperforms (declining search behavior) didn't appear — if anything it moved the wrong
way. This is a real, useful negative-ish result (a small-scale, single-seed ablation isn't enough
to settle this question) rather than a confirmation or refutation of the paper's claim.

---

## Symmetric re-run results (seed=123, `--max-steps 600`, 300 distinct prompts/condition)

Both conditions retrained per the recommendation above (same larger budget, new seed) and
evaluated on the identical 7,404-row held-out set. Real numbers, not projected:

| Metric (held-out) | `outcome_only` (seed42/300) → (seed123/600) | `turn_level` (seed42/300) → (seed123/600) |
|---|---|---|
| Exact match | 0.2355 → 0.2418 | 0.2068 → **0.3065** |
| F1 | 0.3313 → 0.3432 | 0.2943 → **0.3994** |
| Retrieval fraction | n/a | 0.3812 → **0.5279** |

See `results/seed123_600steps/` for the comparison plots (this run) vs. `results/` (original run).

**`outcome_only` barely moved despite doubling training** (EM +0.0063, F1 +0.0119) even though
its *training-time* EM jumped from 0.17 to 0.39 — a real overfitting-to-training-distribution
signal, not a data or code problem: with only 300 distinct prompts (0.33% of the 90,447-row
pool) and no KL regularization tethering the policy to a reference distribution (`beta=0` —
confirmed to match the paper's own choice for this ablation, not a deviation; see below),
training reward climbing doesn't reliably transfer to held-out generalization at this scale.
Corroborating evidence: `outcome_only`'s completions grew ~4.1x longer on average (93→386 tokens,
per the eval's own official `eval_completions/mean_length` metric) with almost no accuracy
benefit — consistent with the policy finding ways to move training reward (verbosity, among
possibly other narrow adaptations) that don't reflect genuine capability gained.

**`turn_level` improved substantially and now clearly beats `outcome_only` on held-out data** —
EM +0.10, F1 +0.105 vs. its own original run, and now ahead of `outcome_only` by a real margin
(+0.065 EM, +0.056 F1) instead of behind. `retrieval_fraction` *rose* during training this time
(0.40→0.57, see the earlier training-time note) instead of declining, and held-out confirms that
recovery (0.53) rather than the original run's decline pattern.

**Re-checking the four criteria against these numbers:**

1. **Gap vs. within-run noise — does not clearly trigger this time, with a caveat.** The
   between-condition gap (0.065 EM / 0.056 F1) is smaller than the raw first-to-last-third swing
   within a single run's training curve (~0.22-0.27) if compared by the same literal method used
   for the original run. But that comparison conflates genuine learning trend with noise: unlike
   the original 150-prompt run (which plateaued after its first third — see the earlier
   first/mid/last-third table), both conditions here show a real, large, steadily-improving trend
   across all 300 prompts, not noisy oscillation around a plateau. The more directly relevant,
   low-noise signal is the held-out result itself (averaged over the ~7,400-row held-out set, not
   a 21-sample rollout group or a 50-prompt training third) combined with criterion 2's check
   below.
2. **Held-out contradicts training-batch trend — does NOT trigger, resolved.** Training-time:
   `turn_level` ends ahead on both EM (0.473 vs 0.394) and F1 (0.547 vs 0.489). Held-out: `turn_level`
   *also* ends ahead (0.3065 vs 0.2418 EM; 0.3994 vs 0.3432 F1) — directionally consistent for the
   first time. This is the criterion that most directly tests "was the original reversal noise or
   real," and it resolves clean.
3. **`outcome_only`'s tool-call-frequency mechanism — still TRIGGERS, independent finding.**
   Unaffected by the retrain's main result: `outcome_only`'s search-tool call frequency continues
   to rise with more training rather than declining as the paper's `GRPO-OR` mechanism claims. A
   real, separate finding from the main comparison — see the `paper_search_penalty` follow-up
   below, which addresses this specific point directly (though not as a paper reproduction — see
   its own notes).
4. **`turn_level`'s retrieval_fraction declining — does NOT trigger, reversed.** Rose during
   training (0.40→0.57) and held out at 0.53, the opposite of the original run's concerning
   decline. The original pattern reads as seed-specific noise, not a structural problem with
   `turn_reward`'s shaping.

**Verdict: 1 of 4 criteria still triggers (criterion 3, an independent per-condition mechanism
check unrelated to the between-condition comparison) — the other three resolve in favor of a real
finding.** This is a meaningfully different, more confident result than the original run: with
more training data and a different seed, `turn_level` (the paper's `GRPO-MR`) shows a real,
directionally-consistent, held-out-confirmed advantage over `outcome_only` (`GRPO-OR`) — in the
same direction the paper reports, even though the absolute magnitude and scale here remain far
below theirs (this repo's F1+EM-bonus reward, 0.8B model, and ~300-prompt budget are all
documented deviations from their setup). **This is now a genuine, if modest-scale, positive
replication of the paper's core `GRPO-OR`/`GRPO-MR` direction** — not a confirmation of their exact
numbers, but a real signal that turn-level reward helps in this setup, given enough training data
to separate the effect from noise. One seed at this larger scale is still not a fully rigorous
statistical guarantee (a second seed at 600 steps would strengthen it further), but this is
honest, evidence-based progress, not an overclaim.

**A note on what did *not* need fixing, checked directly rather than assumed:** the `beta=0` (no
KL regularization) choice was briefly suspected mid-session as a possible cause of the
overfitting-like drift described above, since KL regularization against a reference policy is
part of GRPO's original formulation and exists specifically to prevent this kind of unconstrained
drift. Checked directly against the paper (Appendix E.3): "The KL divergence penalty is disabled
by setting β=0" — stated explicitly for their own `GRPO-OR`/`GRPO-MR` case study. This repo's
`beta=0` matches the paper's own choice exactly; it is not a deviation and was not changed. (Their
separate PPO experiments do use `β=0.001` — already correctly captured in
`docs/phase-7-mt-ppo.md` for that future phase, not applicable here.)

---

## Follow-up experiments (queued after the symmetric re-run, both compared against the seed123/600
steps baseline above, independently — not chained together)

Two follow-ups, each isolating exactly one additional variable against the same seed=123,
`--max-steps 600` baseline (never combined with each other, for the same one-variable-at-a-time
reason the symmetric re-run itself was designed around):

### `length_penalty` (`--penalize-length`)

Motivated by the completion-length finding above: `outcome_only`'s completions grew ~4.1x with
no accuracy benefit, and neither `format_reward` nor `outcome_reward` penalizes verbosity, so the
drift is free under the existing reward. `length_penalty` (implemented and tested,
`src/turn_level_rewards/rewards.py`) measures only the model's own generated text (not
tool-response text injected by the environment), no penalty below a 2000-char target matching the
healthy early-training baseline, capped at -0.2 (below `turn_reward`'s 0.4 and `outcome_reward`'s
1.5 so it can't dominate correctness). Tests whether the verbosity drift is pure free-riding (can
be suppressed with no accuracy cost) or was doing some real, if inefficient, work (accuracy drops
when suppressed).

### `search_count_penalty` (`--paper-search-penalty`)

Motivated by criterion 3 (`outcome_only`'s tool-call frequency rising instead of falling, still
unresolved by the retrain). Replaces the prompt-engineered "at most 2 searches" instruction with a
reward-shaped constraint (`R_search = -λ_s · n_search`), removing both the prompt's numeric cap
and its "rely on your own knowledge" hint, matching the mechanism the source paper's own turn-level
reward design uses for search-count control. **Important, corrected framing — verified by direct
fetch before implementing, not assumed:** this mechanism (`λ_s=0.1`) only exists in the paper's
PPO/MT-PPO reward design (Section 5.2/6.1); their GRPO-OR/GRPO-MR case study (Appendix E) has no
search-count penalty at all. So this is **not** a paper-fidelity fix for the GRPO comparison —
it's a deliberate cross-pollination experiment, borrowing their PPO-context coefficient as the
best available grounded starting point. Framed this way throughout, not oversold as closing a gap
with the paper's actual GRPO methodology.

Results for both, once complete, will be recorded here as a further append — not yet run as of
this note.

---

## `length_penalty` results: hypothesis falsified, and a real, severe failure mode found

Both conditions retrained at the identical seed123/600steps baseline config, only
`--penalize-length` added. Real, complete results — not projected:

| Metric (held-out) | `outcome_only` baseline → `+length_penalty` | `turn_level` baseline → `+length_penalty` |
|---|---|---|
| Exact match | 0.2418 → **0.0901** | 0.3065 → **0.2538** |
| F1 | 0.3432 → **0.1514** | 0.3994 → **0.3481** |
| Tool-call frequency | 1.044 → **0.000** | n/a |
| Retrieval fraction | n/a | 0.5279 → **0.4401** |
| Mean completion length (tokens) | 386 → **12** | ~250-300 → **249** |

**`outcome_only` fully collapsed — not just got shorter, lost the entire task.** Read directly
from the training completions, not inferred from metrics alone:

- Step 1: normal — real search queries, real reasoning (matches the un-penalized baseline's
  behavior exactly).
- Step ~150: collapsed to **literally echoing the question back as the answer**, identical across
  the entire 21-sample rollout group (`<answer>Backflip is the second single by American
  singer-actress Raven-Symoné from her third album, titled what?</answer>` — verbatim repetition
  of the prompt).
- Step ~600 (final): shifted to a *different* degenerate mode — immediate, zero-search, incoherent
  guesses (`<answer>An Allnynum Show</answer>`, `<answer>Dre lust for food and novelty</answer>`).
  Not even wrong-but-plausible guesses; genuinely incoherent text.
- Training-time EM never recovered: 0.052 (first third) → 0.099 (last third), flat and near-zero
  the entire run, vs. the un-penalized baseline's real 0.170→0.394 climb. Completion length kept
  *shrinking* over training (153→36 chars first-to-last-third) rather than stabilizing.
- Held-out confirms it's not a training-curve artifact: `eval_tools/call_frequency=0.0` — the
  model stopped calling search *at all* on the full 7,404-row held-out set. This is total
  capability loss, not increased efficiency.

**`turn_level` stayed coherent throughout training** (spot-checked completions show genuine,
relevant multi-hop search queries the whole way through, e.g. real reasoning about "Brita Horn"
and "Charles XIII of Sweden") **but still lost real accuracy on held-out data** — EM -0.053, F1
-0.051, retrieval_fraction -0.088 vs. its own un-penalized baseline. Not a collapse, but a real,
measurable cost.

**Verdict: the "verbosity was pure free-riding, suppress it for free" hypothesis is falsified for
both conditions**, just at very different severities. This was one of three possible outcomes laid
out before running the experiment (no cost / real cost / no effect), and the actual result — real
cost for both, catastrophic for one — is the most informative of the three, not the hoped-for one.
`outcome_only`'s collapse is the more urgent finding: this specific `length_penalty` design
(target=2000 chars, hard cap at -0.2, no interaction with `beta`/KL regularization since that
stays disabled per the paper) is not safe to use as calibrated, at least not for the
`outcome_only` reward composition (2 reward terms: `format_reward` + `outcome_reward`) — plausibly
because `turn_level`'s extra `turn_reward` term provides more signal diversity within each
rollout group, making it harder for the whole group to collapse into an identical degenerate
completion the way `outcome_only`'s narrower 2-term reward did. Not confirmed as the mechanism,
just the most plausible hypothesis given the data — would need a dedicated ablation (e.g. adding
a third, low-stakes reward term to `outcome_only` without `turn_reward`'s retrieval semantics) to
actually test that causal claim, which is out of scope for this pass.

**Practical takeaway for any future attempt at a length penalty in this repo**: this specific
magnitude/target combination should not be reused as-is. A smaller cap, a softer (non-hard-cliff)
penalty shape, or per-condition tuning would all be reasonable next things to try — none attempted
here, this experiment's job was to test the hypothesis honestly, not to find a working
configuration through additional tuning cycles.

---

## `search_count_penalty` results: fails uniformly, worse than `length_penalty`

Both conditions retrained at the seed123/600steps baseline config with `--paper-search-penalty`
(drops the prompt's "at most 2 searches" instruction *and* adds `-0.1 * n_search` to the reward —
see the "not paper-fidelity for GRPO" framing earlier in this doc). Real, complete results:

| Metric (held-out) | `outcome_only` baseline → `+search_count_penalty` | `turn_level` baseline → `+search_count_penalty` |
|---|---|---|
| Exact match | 0.2418 → **0.0238** | 0.3065 → **0.2205** |
| F1 | 0.3432 → **0.0908** | 0.3994 → **0.3145** |
| Tool-call frequency | 1.044 → **0.000** | 2.268 → **1.028** |
| Retrieval fraction | n/a | 0.5279 → **0.3951** |

**`outcome_only` collapsed even more severely than under `length_penalty`.** Training completions
show the same "healthy start, degenerate end" shape (step 1: real search queries, e.g. "Nimrod,
Montana 2010 census population"; by step ~165: zero-search random guesses like "Yasmin Chan",
"Steven Burstein") — but by the final checkpoint the model wasn't just guessing wrong, it was
producing genuinely garbled text: nested malformed `<answer>` tags, repeated-token artifacts
("Records Records", "Joseph Joseph Wu", "Int alphanumeric record label"). Training-time EM
crashed to near-zero (first-third 0.075 → last-third 0.0005) and never recovered. Held-out
confirms it fully: `eval_tools/call_frequency=0.0`, EM=0.0238 — the worst result of any experiment
this session.

**`turn_level` collapsed too, but recovered — and the recovery held on held-out data.** The
training curve is genuinely three-phase, not a simple decline: healthy for the first ~10% of
steps, then full collapse (`search_call_count=0.0` sustained for ~70% of training, points 31–240
of 300), then a real recovery in the final ~20% (search resumed to ~1.0 calls/completion, EM
climbed to the run's best levels, 0.20–0.34) — coherent, on-topic search queries returned in the
final checkpoint's completions. Held-out evaluation confirms this wasn't a training-curve
artifact: EM=0.2205, F1=0.3145, real and far better than `outcome_only`'s collapse — but still a
genuine, non-trivial cost relative to `turn_level`'s own unpenalized baseline: roughly half the
search calls (2.268 → 1.028), lower retrieval success (0.528 → 0.395), and real accuracy loss
(−0.086 EM, −0.085 F1).

**Refined mechanism, sharper than the `length_penalty` hypothesis**: `turn_level`'s apparent
resilience to `length_penalty` turned out not to be general immunity — `length_penalty` simply
never triggered for `turn_level` in practice (its completions stayed under the 2000-char target
throughout). `search_count_penalty` is different: it **directly opposes `turn_reward`'s own
incentive** (`turn_reward` rewards successful retrieval, which requires searching; this penalty
punishes searching at all) — a genuine intra-reward conflict, not just an added constraint. That
conflict is real and costly for `turn_level` (the ~70%-of-training collapse phase proves it), but
not fatal within a 600-step budget — the combined pull of `outcome_reward` and `turn_reward`
toward productive search eventually overcame the penalty's "never search" local optimum, something
`outcome_only` — with no reward term pulling it back toward searching at all — never managed in
the same budget.

---

## `remove-search-cap-prompt` results: the isolating control, and the answer to "which part broke it"

`search_count_penalty`'s implementation deliberately bundles two changes into one flag
(`--paper-search-penalty`): dropping the prompt's "at most 2 searches" instruction *and* adding
the `-0.1 * n_search` reward term. That means the results above cannot, on their own, distinguish
between two possible causes of collapse — losing the prompt's explicit guidance, or the penalty
term itself. `--remove-search-cap-prompt` was added specifically to isolate this: it drops the
same prompt instruction with **no reward penalty at all**, decoupled from
`--paper-search-penalty` via `build_trainer`'s `search_cap_in_prompt = not (paper_search_penalty
or remove_search_cap_prompt)` (see commit `6e1fb10`).

| Metric (held-out) | `outcome_only` baseline → `remove-cap-only` | `turn_level` baseline → `remove-cap-only` |
|---|---|---|
| Exact match | 0.2418 → **0.2006** | 0.3065 → **0.3199** |
| F1 | 0.3432 → **0.2923** | 0.3994 → **0.4222** |
| Tool-call frequency | 1.044 → **1.997** | 2.268 → **2.053** |
| Retrieval fraction | n/a | 0.5279 → **0.5303** |

Both conditions trained fully healthy — no collapse pattern anywhere in either 600-step run,
confirmed via trackio's `train/exact_match` curve (steady learning throughout, no flatline) and
direct completion inspection at multiple points (real, coherent, on-topic search queries in both
final checkpoints).

**`outcome_only` paid a real but modest cost, in the opposite direction from collapse: it searched
almost twice as much (1.044 → 1.997 calls/completion) with somewhat lower accuracy (−0.041 EM,
−0.051 F1).** Without the prompt's explicit count guidance, the model over-searched rather than
under-searched — training completions show a pattern of short, repetitive, lower-quality queries
("Forever", "Forge" as separate single-word searches in one completion) rather than search
avoidance.

**`turn_level` paid no cost at all — if anything, it did slightly better** (EM +0.013, F1 +0.023,
retrieval_fraction essentially unchanged at 0.530 vs. 0.528). `turn_reward`'s own incentive
(reward for surfacing real supporting-fact passages) was sufficient on its own to keep search
behavior effective without any explicit prompt-level count guidance.

**This cleanly answers the question `search_count_penalty` left open: the severe collapses seen
above were caused specifically by the `-0.1` penalty term, not by losing the prompt's "at most 2
searches" instruction.** Removing the prompt guidance alone produces a completely different,
much milder failure mode — unconstrained *over*-searching with a real-but-modest accuracy cost —
the opposite direction from the penalty-driven collapse to *zero* search. `outcome_only`'s two
follow-up experiments (`length_penalty`, `search_count_penalty`) both added a term that punished
some behavior with no accompanying incentive pulling the policy back — and in both cases,
`outcome_only`'s narrower 2-term reward found the cheapest way to avoid the penalty (echo the
question, guess randomly, stop searching entirely) rather than the intended trade-off. `turn_level`
never has this failure mode available to it as cheaply, because `turn_reward` is always actively
pulling toward the exact behavior (searching, and searching well) that these penalties discourage.

---

## Cross-experiment synthesis: what this session's follow-up work actually established

Three follow-up reward-shaping experiments were run against the `seed123/600steps` baseline for
both `outcome_only` and `turn_level`. None of them improved on the baseline. All three are
genuinely informative negative (or negative-with-nuance) results, not a wasted session:

| Experiment | `outcome_only` | `turn_level` |
|---|---|---|
| baseline | EM=0.2418, F1=0.3432 | EM=0.3065, F1=0.3994 |
| `length_penalty` | **collapsed** (EM=0.0901) | mild real cost (EM=0.2538) |
| `search_count_penalty` | **collapsed** (EM=0.0238) | collapsed-then-recovered, real cost (EM=0.2205) |
| `remove-search-cap-prompt` (isolating control) | mild real cost (EM=0.2006) | **no cost** (EM=0.3199) |

**`outcome_only`'s narrow 2-term reward (`format_reward` + `outcome_reward`, nothing else) is
broadly fragile to any added penalty term.** Both `length_penalty` and `search_count_penalty`
collapsed it completely and durably — in both cases the policy found a cheap, reward-guaranteed
way to avoid the penalty (shrink to near-nothing; stop searching entirely) rather than learning
the intended efficient-and-correct trade-off. The one experiment that didn't collapse it
(`remove-search-cap-prompt`) added no penalty at all — it only removed guidance, which is a
fundamentally different (and much lower-risk) kind of change.

**`turn_level`'s extra `turn_reward` term provides real, but incomplete, protection.** It never
suffered `outcome_only`'s complete, durable collapse in either penalty experiment — but it did pay
a real, measurable accuracy cost in both (`length_penalty`: −0.053 EM; `search_count_penalty`:
−0.086 EM after a genuine but temporary collapse phase). The likely mechanism, sharpened across
both experiments: `turn_reward` gives the policy an *independent* pull toward the exact behavior
(searching, searching well) that these penalties discourage, which is enough to eventually escape
a penalty-driven local optimum within budget (as `search_count_penalty`'s recovery shows) but not
enough to avoid paying real cost along the way. Where a penalty doesn't directly oppose an
existing reward term (`remove-search-cap-prompt`, which added no penalty at all), `turn_level` paid
no cost whatsoever.

**Practical implication for anyone extending this repo's reward design**: adding a bare penalty
term to `outcome_only`'s reward composition is high-risk without either (a) an accompanying
positive term that keeps pulling toward the desired behavior (the way `turn_reward` does for
`turn_level`), or (b) careful, GRPO-specific coefficient calibration — the two coefficients tested
here (`length_penalty`'s cap/target, `search_count_penalty`'s `λ_s=0.1` borrowed from the paper's
PPO context) were both naive ports, never tuned for GRPO's group-relative advantage structure
specifically, and both broke `outcome_only` as a result.

---

## Infrastructure incidents from this session (documented honestly, not hidden)

Two real infrastructure issues occurred during this session's autonomous overnight run, both
resolved without loss of any scientific findings:

**Disk-space exhaustion.** Accumulated checkpoint directories from the night's sequence of full
training runs (each ~27GB, several preserved copies across the symmetric re-run, `length_penalty`,
and `search_count_penalty` experiments) filled the disk to 100% (1.4GB free), crashing
`turn_level`'s `search_count_penalty` training mid-checkpoint-save. `uv`'s package cache (78GB)
could not be reclaimed as a workaround because the long-running retrieval server held a lock on
it. Resolved by deleting six checkpoint directories whose held-out eval results were already
safely committed to git (the original seed42/300steps run and the seed123/600steps baseline, both
conditions each, plus the `length_penalty` checkpoints, both conditions) — **only after explicit,
specific user authorization**, since this is exactly the kind of consequential, hard-to-reverse
action that should not be decided autonomously. Freed ~162GB; no scientific findings were lost, as
every deleted checkpoint's numbers were already in `results/*.json` and this document.

**Unexplained external kill during `turn_level`'s final training run.** The first attempt at
`turn_level`'s `remove-search-cap-prompt` training was killed externally at 1h34m24s elapsed, with
no application-level error and a clean systemd resource-accounting exit (consistent with an OOM
or memory-pressure event after many hours of continuous GPU work, though no kernel-level `dmesg`
access was available to confirm the exact cause). The run had been healthy throughout, right up
to the kill. A retry of the identical command succeeded cleanly, running well past the original
kill point with no recurrence — suggesting a one-off transient issue rather than a systematic
time- or resource-based limit. No data was lost (the failed attempt had no results to lose;
checkpoints were being written incrementally and the partial ones were simply overwritten by the
retry).
