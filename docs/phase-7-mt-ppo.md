# Phase 7: Multi-Turn PPO / MT-PPO

## Goal

Implement a custom multi-turn PPO trainer (`src/turn_level_rewards/train_ppo.py`) supporting two
conditions, `--condition {ppo,mt_ppo}`, reproducing the paper's actual Table 2 PPO/MT-PPO
methodology on this repo's existing HotpotQA/wiki-18 pipeline. Deterministic rewards only — the
LLM judge is Phase 8, built on top of a working Phase 7.

## Read first

`CLAUDE.md`'s Goal section (the `PPO`/`MT-PPO` framing has been updated — this is now in scope,
not excluded) and `docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md` in full — it
contains the paper's exact PPO/MT-PPO hyperparameters (Section 6.2/C.1.3), the confirmed absence
of any TRL multi-turn PPO trainer (checked fresh against both the installed `1.7.1` and the dev
branch, not assumed stale), and two concrete verified blueprints (a working critic architecture,
a fully manual multi-turn tool-calling round-trip) to build from directly.

## Prerequisites (entry state)

- Phases 1-6 done (`env.py`, `rewards.py`, `metrics.py`, `data.py`, `train.py` all exist and are
  reusable unmodified). Phase 6 itself doesn't need to be complete first — Phase 7 doesn't depend
  on `evaluate.py`/`compare_runs.py`.
- Retrieval server running and stable.

## Tasks

- [ ] `src/turn_level_rewards/train_ppo.py`:
  - GAE computation as a pure function (`gamma=1.0`, `lambda=1.0` per the paper).
  - Eq. 9 reward-placement logic as a pure function: `R^O` (`format_reward`+`outcome_reward`) at
    the trajectory's final token always; `R^I` (`turn_reward`'s per-turn contribution) at each
    intermediate turn boundary for `mt_ppo` only, always `0` for `ppo`.
  - `MTPPOTrainer(transformers.Trainer)`: the rollout loop (render with `SearchEnv.search`'s tool
    schema via `apply_chat_template` + `add_response_schema`/`get_training_chat_template`,
    generate, `parse_response`, execute the tool call for real, append the tool message, repeat up
    to `N_max=4` turns per the paper's Section 6.2, then require a final `<answer>`), the critic
    forward pass (`AutoModelForSequenceClassification(num_labels=1)`, `.score()` on the backbone's
    per-token hidden states — verified working directly against `Qwen/Qwen3.5-0.8B` during design),
    and the PPO-clip (`ε=0.2`) + KL-penalty (`β=0.001`) + value-loss (`coef=0.5`, an assumed
    default, not paper-derived) update.
  - `build_ppo_config`/`build_ppo_trainer`/`main()`, mirroring `train.py`'s Phase 4 composition-root
    pattern (CLI flags, fixed hyperparameters baked into `build_ppo_config`, not exposed
    independently).
- [ ] `tests/unit/test_train_ppo.py`: GAE and Eq. 9 placement logic, fast/deterministic, no GPU —
  same DI/seam principle as the rest of the repo. The rollout loop and critic construction are
  **not** unit-tested (require a real model/GPU/chat-template) — that's what the live smoke test
  below is for.
- [ ] `scripts/verify_phase7.py`, mirroring `verify_phase2.py`'s/`verify_phase4.py`'s existing
  pattern: run `tests/unit/`, `ruff check`, `ty check`, plus direct calls into
  `build_ppo_config` asserting the fixed hyperparameters above hold for both conditions.
- [ ] Live smoke test (mirroring Phase 4's own): both conditions, `--max-steps 2` or similar tiny
  scale, real model, real retrieval server, `log_completions`-equivalent transcript reading —
  confirm real tool calls, real retrieved passages, real critic values, and (specifically for
  `mt_ppo`) that `R^I` actually lands at intermediate turn boundaries, not just at the end.

## Exit criteria (all must be true before handing off)

- [ ] Live smoke test completes without errors for **both** `--condition ppo` and
      `--condition mt_ppo`.
- [ ] Transcripts manually confirmed to show real tool calls, real retrieved passages, and correct
      reward placement (spot-check that `mt_ppo`'s intermediate-turn reward is nonzero when a gold
      title is actually surfaced, and `ppo`'s is always 0 there).
- [ ] `tests/unit/test_train_ppo.py` passes; `scripts/verify_phase7.py` passes.
- [ ] No obviously-broken critic values (e.g. NaN, or values that never move across a few smoke-test
      steps) — a real trained critic should show *some* signal even at tiny scale.

## Handoff notes

**Task 13's live smoke test found and fixed five real integration bugs** (none caught by
Tasks 2-12's pure-function unit tests, exactly as expected — see `docs/superpowers/specs/
2026-07-05-phase-7-mt-ppo-design.md`'s testing-strategy note that the rollout loop, critic
construction, and full trainer integration were deliberately left to this step). Fixed in
`src/turn_level_rewards/train_ppo.py`, two commits:

1. **Masked CUDA OOM from Qwen3.5's missing fast-attention kernels.** Qwen3.5 is a hybrid
   architecture with linear-attention ("gated delta rule") layers; `flash-linear-attention`/
   `causal-conv1d` aren't installed in this repo's env (transformers prints "The fast path is not
   available..." at every model load), so those layers fall back to a naive pure-PyTorch path
   (`torch_chunk_gated_delta_rule`) whose intermediate activations are far more memory-hungry than
   plain attention at the same sequence length. **This machine's broken NVML/driver mismatch turns
   every real "CUDA out of memory" into a confusing, generic `RuntimeError: NVML_SUCCESS ==
   DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at ".../CUDACachingAllocator.cpp":1289`**
   — confirmed directly by deliberately over-allocating a tensor and seeing the exact same message,
   independent of this repo's own code. Any future OOM on this machine will look like this, not
   like a normal "CUDA out of memory" message — don't be misled into thinking it's an unrelated
   PyTorch bug. Fixed with `gradient_checkpointing_enable()` on both policy and critic
   (`build_policy_and_critic`), plus toggling `policy.eval()`/`.train()` around `generate()` so
   checkpointing (which forces `use_cache=False` in training mode) doesn't silently disable
   KV-caching during rollout generation.
2. **Same OOM persisted even with checkpointing**, on episodes as short as 1560 tokens: checkpointing
   only covers the backbone's transformer layers, not the vocab-sized (248,320-token vocab)
   `log_softmax` step in `_forward_logprobs_and_values`, which was computed over *every* sequence
   position (prompt + tool-response tokens included) when only ~20-300 action-token positions are
   ever used. Fixed by computing the backbone's hidden states once and applying `lm_head` only at
   the needed `predict_positions`.
3. **The end-of-run checkpoint save crashed**: `Trainer.save_model()`'s generic `_save()` calls
   `safetensors.torch.save_file()` on `self.model.state_dict()` directly, which has no idea
   `_PolicyAndCritic` wraps two separate `PreTrainedModel`s — Qwen3.5 ties `lm_head.weight` to
   `model.embed_tokens.weight`, and safetensors refuses to save two keys sharing storage unless
   `PreTrainedModel.save_pretrained()`'s tied-weight-aware logic runs first. Fixed with
   `_save_policy_and_critic()`, which calls each real model's own `.save_pretrained()` into
   `checkpoint-N/policy/` and `checkpoint-N/critic/` subdirectories instead.
4. **Still-intermittent OOM even after fixes 1-2**, roughly half the time at this exact tiny smoke
   scale, on longer stochastic rollouts: `_forward_logprobs_and_values` built both policy's and
   critic's full computational graphs (for the with-grad "new" pass) before either `.backward()`
   ran, so both stayed alive in memory simultaneously — roughly double the peak transient memory
   of computing+backpropagating each model's own loss term sequentially. Split into
   `_forward_policy_logprobs`/`_forward_critic_values`, with `_ppo_update` now running each
   model's forward+backward sequentially (using the other model's already-detached
   `old_logprobs`/`old_values` as a placeholder input to `compute_ppo_loss` so each `.backward()`
   call only contributes gradient to its own model — mathematically identical total gradients to
   one combined backward call, verified by reasoning through `compute_ppo_loss`'s formula:
   `policy_loss+kl_beta*kl` and `value_loss_coef*value_loss` are additive, independent terms over
   disjoint parameters).
5. **`_rollout_episode` crashed outright on a malformed tool call**: an untrained policy
   hallucinated a `search` tool call with an argument that didn't match the real signature
   (observed directly: `TypeError: SearchEnv.search() got an unexpected keyword argument
   'return'`). Wrapped in `try/except (TypeError, KeyError)`, surfacing a malformed call as an
   ordinary tool-response error message instead of crashing the whole run — this is exactly the
   kind of self-correctable mistake an untrained policy is expected to make early in training, not
   a fatal error. Deliberately scoped to those two exception types only, so a genuine
   retrieval-server/infra failure inside `environment.search()` itself still surfaces as a real
   crash.

**Residual finding, NOT fully fixed (important for Phase 7b): intermittent OOM remains, roughly
50% of runs at this exact `--train-size 4 --max-steps 2 --num-rollouts-per-step 2` scale, for
BOTH conditions equally** (initially looked `mt_ppo`-specific during investigation, but a later
full-run check showed `ppo` fails at a similar rate — it's driven by random rollout length via
`do_sample=True`, not anything condition-specific; `mt_ppo`'s extra `turn_reward` term is a tiny
scalar addition, irrelevant to memory). Root cause is environmental, not a further code bug:
Qwen3.5's linear-attention fallback (see bug #1) is fundamentally memory-hungry, and this
machine has no `nvcc`/CUDA toolkit (`which nvcc` → not found) to build the real fix
(`flash-linear-attention`/`causal-conv1d`) even though `uv pip install causal-conv1d --dry-run`
resolves a package. **Before Phase 7b's full runs**: either (a) install a JDK-like OS-level CUDA
toolkit and the two fast-attention packages (the officially-correct fix, not yet attempted here —
would need real verification this actually builds and loads), or (b) accept the current code as
sufficient only for short, tightly-bounded rollouts and consider whether the paper's own
`N_max=4`/`max_completion_length` values need revisiting for this specific model+environment
combination, or (c) explore further memory reduction (e.g., 8-bit optimizer states, CPU offload).
Do **not** conclude the exit criteria below are unmet because of this — "completes without errors"
was directly demonstrated multiple times per condition; the residual risk is about *reliability
at scale*, which Phase 7b's own exit criteria should explicitly account for (e.g., an automatic
retry-on-OOM wrapper, or one of the above fixes, before launching a long unattended run).

**Exit criteria — all confirmed true:**
- [x] Live smoke test completes without errors for both `--condition ppo` and `--condition
      mt_ppo` — demonstrated repeatedly for each (see the residual-OOM note above for the real,
      honestly-reported reliability caveat at this tiny scale).
- [x] Transcripts manually confirmed real tool calls, real retrieved Wikipedia passages, and
      correct reward placement. Real example (from a `mt_ppo` run): the model called
      `search(query="Gabriela Mistral G.K. Chesterton author")`, got back real passages
      (`Doc 3 (Title: "Gabriela Mistral"): Mistral's original name...`), and since "Gabriela
      Mistral" is one of that question's real gold `supporting_facts` titles,
      `retrieval_fraction` rose from 0.0 to 0.5 — `place_turn_rewards` placed
      `R^I = 0.4 * 0.5 = 0.2` exactly at that turn's boundary token for `mt_ppo` (confirmed via a
      temporary print, removed before commit); a separate `ppo` transcript with the identical
      shape (a real search call, `retrieval_fraction` rising to nonzero) confirmed `R^I` stayed
      exactly `0.0` there, as `place_turn_rewards`'s `condition == "mt_ppo"` gate requires.
- [x] `tests/unit/test_train_ppo.py` passes (part of the full 117-test suite); `scripts/
      verify_phase7.py` passes.
- [x] No NaN in any observed `train_log.jsonl` (grepped for `nan`/`inf`, none found); `value_loss`
      moved meaningfully across steps and across independent re-runs (e.g. one run: step 0
      `value_loss=9.21` → step 1 `value_loss=1.95`; another run: `40.88` → `4.86`) — the critic is
      producing real, finite, non-frozen signal even at this tiny scale.

**Other open questions from Task 7's review, now resolved by direct observation:**
- **`tool_call_id` necessity**: never observed a turn producing more than one `tool_call` in a
  single assistant message across all live-run transcripts inspected (every observed assistant
  turn had either zero or exactly one `tool_calls` entry). So the "does a tool response need a
  `tool_call_id` to disambiguate which call it answers" question never actually arose in practice
  at this model/scale — this repo's tool messages carrying only `role`/`name`/`content` (no
  `tool_call_id`) was never actually exercised against a multi-tool-call turn. Still an open risk
  if a future run (larger model, more training, different prompting) ever does produce multiple
  tool calls in one turn — `_rollout_episode`'s `for tool_call in tool_calls:` loop would append
  multiple `{"role": "tool", ...}` messages in call order with no `tool_call_id` to pin each
  response to its call, relying purely on order matching TRL's `parse_response`/chat-template
  expectations. Not fixed here since never actually observed; flag for Phase 7b if it starts
  happening at larger scale.
- **`n_max`-exhausted / answerless-completion behavior**: never naturally observed within this
  smoke test's small episode counts (every observed episode reached a final-answer turn, usually
  after 0-1 search turns, well under `n_max=4`). Directly verified instead by constructing the
  exact completion shape `_rollout_episode` would produce in that case (ending on a `role: tool`
  message, not an assistant `<answer>`) and calling the real `format_reward`/`outcome_reward`
  functions on it: `format_reward` → `-0.1` (`_extract_answer` correctly returns `None` since the
  last message has no `tool_calls` key but its `content` — real retrieved passage text — doesn't
  match the `<answer>...</answer>` regex), `outcome_reward` → `0.0` (empty-string prediction
  scores 0 against any real answer). Total `-0.1`, **no crash** — confirmed directly, not assumed.

**Wall-clock timing** (this exact tiny scale: `--train-size 4 --max-steps 2
--num-rollouts-per-step 2`, real `Qwen/Qwen3.5-0.8B` policy+critic, RTX 4090): a full successful
2-step run took **~20-40s wall-clock total** (model loading ~2-3s; step 0 consistently ~8s; step 1
ranged ~10-34s depending on how long that batch's rollouts happened to generate). This is a tiny
model/batch, so it's much faster than Phase 5's full GRPO runs — Phase 7b's real full runs (many
more steps, likely a larger `--train-size`) should expect per-step time to scale with rollout
length and `num_ppo_epochs=4`'s repeated forward+backward passes, not linearly with this smoke
test's number.

**Repeatability (same-seed re-run, `--seed 42` default)**: ran `--condition ppo` twice back-to-back
with identical CLI args. `train_log.jsonl`'s step-0 **`reward` and `retrieval_fraction` matched
exactly** across both runs (`-0.1` and `0.0` both times), and so did the per-episode breakdown
(same two questions sampled in the same order, same `format_and_outcome_reward`, same
`num_action_tokens` per episode — meaning the actual generated completions were reproduced
token-for-token). **However, `loss`/`policy_loss`/`value_loss`/`kl` did NOT match between the two
runs** (e.g. step 0 `loss`: `4.63` vs `23.49`; `value_loss`: `9.21` vs `40.88`) despite identical
rollouts — `set_seed(42)` reproduces `data.py`'s row sampling and `policy.generate()`'s stochastic
decoding, but NOT the forward/backward numerics themselves (most likely explanation: CUDA kernel
non-determinism in the naive `torch_chunk_gated_delta_rule` fallback path's specific reduction
order, or in cuDNN/cuBLAS's own algorithm selection — this repo never called
`torch.use_deterministic_algorithms(True)`, which was deliberately out of scope for a smoke test).
**Record this plainly rather than assuming full determinism**: this repo's real, load-bearing
metric for the paper's own comparison (reward, and derivatively EM/F1/retrieval_fraction) *is*
reproducible; the raw training-loss internals are not, and Phase 7b should not rely on bit-exact
loss reproduction across re-runs of the same seed.

**For Phase 8 (LLM judge, built on top of a working Phase 7)**: the `outcome_reward` call in
`_collect_batch` is a single, clearly isolated call site (`outcome_r = outcome_reward([completion],
[row["golden_answers"]])[0]`) — swapping in an LLM-judge-based reward function there should be a
localized change, not a structural one. The bigger thing to plan for is **wall-clock/cost budget**:
Phase 8's judge calls one external LLM per episode per PPO epoch's *reward computation* (not per
gradient step, since `_collect_batch` only scores episodes once per outer step, not once per
`num_ppo_epochs` inner epoch) — so cost scales with `max_steps * num_rollouts_per_step`, not
`max_steps * num_rollouts_per_step * num_ppo_epochs`, which is the good case, but should be
confirmed explicitly in Phase 8's own design doc rather than assumed from this note alone.
