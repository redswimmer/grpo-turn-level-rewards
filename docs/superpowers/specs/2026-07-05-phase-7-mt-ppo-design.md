# Phase 7: Multi-Turn PPO / MT-PPO — Design

Status: approved by user, 2026-07-05. Implements `docs/phase-7-mt-ppo.md`.

## Context and evidence gathered

This design was grounded by fetching the paper's actual PPO/MT-PPO methodology directly (not
assumed from the GRPO ablation's numbers, which are a separate experimental setup), by re-checking
TRL's *current* installed and dev-branch source for multi-turn PPO support (not trusting
CLAUDE.md's older "out of scope" note to still be accurate), and by manually executing both the
critic architecture and the full multi-turn tool-calling loop against the real model and the real
retrieval server — proving the two riskiest assumptions with real code, not documentation reading.

**TRL still has no multi-turn PPO trainer, confirmed fresh, not stale:**
- `trl.experimental.ppo.PPOTrainer` (moved out of top-level `trl` in the currently-installed
  `1.7.1`) takes a real, separate `value_model: PreTrainedModel` — a genuine critic, matching the
  paper's method — but has zero `environment_factory`/tool-calling support (confirmed by grep,
  zero matches).
- Checked the *dev branch* (`/home/asavala/Development/trl`, commit `b73bc7b9`, dated 2026-07-05 —
  today): recent commit history shows tool-calling support was explicitly added to `GRPOTrainer`,
  `KTOTrainer` ("Align KTO with DPO: Support tool calling"), and `RLOOTrainer` ("Add tool support
  to RLOOTrainer") — but never to `PPOTrainer`. This reads as a deliberate architectural boundary
  (PPO's critic/GAE structure doesn't fit the same rollout-pooling pattern the other trainers use),
  not a "coming soon" gap.
- `DPPOTrainer` (new experimental variant, subclasses `GRPOTrainer`) does inherit multi-turn
  tool-calling for free, but it only swaps GRPO's clip-mask mechanism (a "principled trust region"
  alternative) — no critic, no value head, no GAE. Not a viable shortcut for adding a real critic;
  inheriting from `GRPOTrainer` to bolt on a critic+GAE would mean overriding far more internals
  than DPPO's precedent touches, with the same fragility CLAUDE.md already flagged for `MT-GRPO`.
- **Decision: build `MTPPOTrainer` directly on `transformers.Trainer`**, not subclassing
  `GRPOTrainer` or `PPOTrainer`. More code upfront; every piece is ours, explainable, and not
  coupled to internals TRL didn't design as an extension point.

**Paper's actual PPO/MT-PPO methodology (Section 6.2, C.1.3), fetched directly — distinct from the
GRPO ablation's own separate setup (Appendix E.3, already used for Phase 5)**:
- Max turns: `N_max = 4` (Section 6.2) — this repo's own separate value for Phase 7, not the same
  number as the GRPO conditions' `max_tool_calling_iterations=4` (that one is a safety-net cutoff
  above a smaller soft "2 searches" limit; here `N_max=4` is the paper's own primary spec).
- **Eq. 9, turn-boundary reward placement**: `r_t = R^O` at the last token of the whole trajectory,
  `r_t = R^I` at the last token of each intermediate turn, `0` elsewhere. This is the mechanism
  that distinguishes plain `ppo` from `mt_ppo` in this design (see "Two conditions" below) — and
  it's *why* MT-PPO is tractable here while `MT-GRPO` remains out of scope: PPO already has a
  real per-token critic and GAE, so turn-level credit assignment falls out of reward placement
  alone, without needing `MT-GRPO`'s separate extra-rollout advantage trick (Eq. 5, Appendix D).
- **GAE**: standard formula (`A_t = Σ(γλ)^l δ_{t+l}`, `δ_t = r_t + γV_{t+1} - V_t`), with
  **γ=1, λ=1** — this simplifies things: at these values GAE reduces toward a full-episode
  Monte-Carlo-return-minus-baseline, no discount/decay tuning needed.
- **Critic**: a separate model, not a shared backbone with the policy. Policy LR = `1e-6`, critic
  LR = `1e-5` (10x higher). Clip `ε = 0.2`, KL penalty `β = 0.001` (both a clip *and* a direct KL
  term, not either/or — standard PPO practice interprets this KL as against the rollout-time
  policy, i.e. the trust-region reference, not a separate frozen reference model the way GRPO's
  `beta` works; the paper doesn't elaborate further, so this is a stated assumption, not a
  paper-derived detail).
- **Training structure**: 4 inner update epochs per collected rollout batch, over 500 total
  rollout-collection steps.
- **Value loss coefficient**: not given by the paper. Uses the standard PPO-literature default of
  `0.5`, documented here as an assumed default, not a paper number — flag this explicitly rather
  than implying it was derived.
- **Dataset**: paper's Table 2 evaluates across six datasets including HotpotQA (marked
  in-domain), confirming this repo's existing `data.py`/`SearchEnv`/wiki-18 pipeline is directly
  reusable — no new dataset plumbing needed for Phase 7.
- **Reward source**: Table 2's headline numbers use **deterministic/verifiable rewards only** —
  the LLM-as-judge scheme (Appendix C.2/C.3, `gpt-oss-120b`) is a separate exploratory variant
  shown only on NQ (Figure 6), not what drove the main comparison. This is why the judge is
  Phase 8, not folded into Phase 7.

**Critic architecture, verified by actually instantiating it against `Qwen/Qwen3.5-0.8B`** (not
assumed from TRL's docstrings):
```python
critic = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3.5-0.8B", num_labels=1, dtype=torch.bfloat16
)
```
This cleanly instantiates `Qwen3_5ForSequenceClassification` (confirmed no architecture-specific
failure from Qwen3.5's hybrid linear-attention layers) with a real `.score()` method. Confirmed
that calling `.score()` directly on the *backbone's* full per-token `hidden_states[-1]` (not the
pooled classification-head forward most callers use) gives a genuine **per-token** value estimate
— shape `[batch, seq_len, 1]`, one scalar per token position, exactly matching TRL's own
`PolicyAndValueWrapper` pattern (`critic_backbone(**kwargs)` then
`value_model.score(hidden_states[-1])`) and exactly what GAE needs at each turn boundary. Verified
by direct execution, not read from source alone.

**Multi-turn tool-calling loop, verified by manually driving one full round-trip** with zero
`GRPOTrainer` involvement — this is the concrete blueprint for `MTPPOTrainer`'s rollout loop:
1. `tok = add_response_schema(AutoTokenizer.from_pretrained(...))`, then
   `training_template = get_training_chat_template(tok)` — **a genuinely new, undocumented-until-now
   requirement**: `parse_response` silently needs these applied first (confirmed by hitting
   `AttributeError: This tokenizer does not have a response_template...` without them).
   `get_training_chat_template`'s docstring explicitly lists Qwen3.5 as supported.
2. `tok.apply_chat_template(messages, tools=[get_json_schema(SearchEnv.search)],
   add_generation_prompt=True, chat_template=training_template, ...)` → `model.generate(...)`.
3. `parse_response(tok, new_token_ids, prefix=prompt_token_ids)` — correctly parsed a real
   `tool_calls` list (`search(query="Corliss Archer film Kiss and Tell")`) from a real generation.
4. Executed the **real** `SearchEnv.search()` against the live retrieval server — got a real
   "Kiss and Tell (1945 film)" passage back; `retrieval_fraction` correctly updated to `1.0`.
5. Appended the assistant (with `tool_calls`) and `tool` messages manually, re-rendered, generated
   again: no further tool calls, a real natural-language answer correctly wrapped in
   `<answer>...</answer>`.

All of `apply_chat_template`, `add_response_schema`, `get_training_chat_template`, and
`parse_response` are public functions in `trl.chat_template_utils` — legitimate, documented
building blocks, not private `GRPOTrainer` internals being reused improperly.

## Goal

Implement `src/turn_level_rewards/train_ppo.py`: a from-scratch `MTPPOTrainer` supporting two
conditions, `ppo` and `mt_ppo`, reusing `SearchEnv`/`rewards.py`/`data.py` unmodified. Matches the
paper's actual Table 2 PPO/MT-PPO methodology (deterministic rewards only — the LLM judge is
Phase 8, built on top of a working Phase 7, not part of this phase).

## Two conditions (`ppo` vs `mt_ppo`)

Mirrors this repo's existing `outcome_only`/`turn_level` pairing structure exactly — same reward
functions in both conditions, differing only in **where** reward is placed in the token sequence
(Eq. 9), not which reward functions run:

| | `ppo` | `mt_ppo` |
|---|---|---|
| `R^O` (= `format_reward` + `outcome_reward`, summed) | at the last token of the whole trajectory | at the last token of the whole trajectory |
| `R^I` (= `turn_reward`'s per-turn contribution) | **always 0** — all credit collapses to the final token, single lump-sum credit assignment even across a multi-turn episode | placed at the last token of **each intermediate turn** — real per-turn credit assignment via the critic's GAE bootstrapping between turns |

This is a genuinely fair pair: identical reward functions, identical model, identical rollout
mechanics — the only variable is reward *placement*, which is precisely the algorithmic
difference Eq. 9 encodes between "PPO" and "MT-PPO" in the paper.

## Module structure (`src/turn_level_rewards/train_ppo.py`)

```python
class MTPPOTrainer(transformers.Trainer):
    """Custom multi-turn PPO trainer with tool-calling, built directly on transformers.Trainer
    (not GRPOTrainer/PPOTrainer -- see design spec's Context section for why).

    Owns: the rollout loop (render with tools -> generate -> parse_response -> execute
    SearchEnv.search() on a tool call -> append tool message -> repeat up to N_max=4 turns ->
    require a final <answer>), the critic forward pass, Eq. 9 reward placement, GAE (gamma=1,
    lambda=1), and the PPO-clip + KL-penalty + value-loss update.
    """

def build_ppo_config(condition: Literal["ppo", "mt_ppo"], seed: int, ...) -> MTPPOConfig:
    """Pure function, mirrors build_config's role from Phase 4. Fixed: N_max=4, clip=0.2,
    kl_beta=0.001, policy_lr=1e-6, critic_lr=1e-5, gamma=1.0, gae_lambda=1.0,
    num_ppo_epochs=4, value_loss_coef=0.5 (assumed default, not paper-derived -- see Context)."""

def build_ppo_trainer(condition, config) -> MTPPOTrainer:
    """Composition root: real policy model, real critic
    (AutoModelForSequenceClassification(num_labels=1)), real SearchEnv, real data. Not
    unit-tested -- validated by the live smoke test instead, same principle as Phase 4's
    build_trainer."""

def main() -> None:
    """CLI entrypoint: --condition {ppo,mt_ppo}, plus size/step flags mirroring train.py's
    pattern from Phase 4."""
```

## Testing strategy (same DI/seam principles as the rest of the repo)

- **GAE computation**: pure function taking reward/value sequences, returns advantages. Unit-test
  with fake sequences, including a hand-computed expected result at `gamma=1, lambda=1`'s
  simplified case.
- **Eq. 9 reward-placement logic**: pure function mapping `(turn boundary token indices,
  format_reward, outcome_reward, turn_reward, condition)` → per-token reward array. Unit-test with
  fake trajectory structures (no real model), asserting `ppo`'s `R^I` is always 0 and `mt_ppo`'s
  isn't.
- **The rollout loop and critic construction are NOT unit-tested** — same principle as Phase 4's
  `build_trainer`: they require a real model/GPU/chat-template, which is exactly what a live smoke
  test (mirroring Phase 4's own) validates instead, not `tests/unit/`.

## Out of scope for Phase 7

- **The LLM judge** (Phase 8) — deliberately deferred; the paper's own headline Table 2 numbers
  don't use it either.
- **`MT-GRPO`** — still genuinely out of scope, unchanged from CLAUDE.md's existing reasoning:
  `GRPOTrainer` has no supported hook to override its built-in advantage computation with a custom
  per-turn one. This is unrelated to the PPO work above and isn't reopened by it.
- vLLM / distributed training — not needed at this repo's single-GPU scale, same as Phases 1-6.
