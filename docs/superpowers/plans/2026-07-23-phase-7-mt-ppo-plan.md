# Phase 7: Multi-Turn PPO / MT-PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/turn_level_rewards/train_ppo.py`: a from-scratch multi-turn PPO trainer
(`MTPPOTrainer`) supporting `--condition {ppo,mt_ppo}`, reusing `SearchEnv`/`rewards.py`/`data.py`
unmodified, matching the paper's Table 2 PPO/MT-PPO methodology (Eq. 9 turn-boundary reward
placement, GAE, PPO-clip + KL + value loss). Deterministic rewards only — this is Phase 7 exactly
as scoped in `docs/phase-7-mt-ppo.md` (build + smoke test); full training runs are Phase 7b.

**Architecture:** `MTPPOTrainer` subclasses `transformers.Trainer` for its checkpoint/logging/state
plumbing only — it overrides `train()` entirely with a custom outer loop (collect a batch of
episodes via a hand-built multi-turn tool-calling rollout loop → compute rewards/GAE → run
`num_ppo_epochs` inner gradient-accumulated update passes), since PPO's rollout-then-multi-epoch
structure doesn't fit `Trainer`'s default single-pass-per-batch loop. Policy and critic are two
separate models wrapped in one `nn.Module` so `Trainer`'s standard plumbing sees one `self.model`,
with `create_optimizer` giving them independent learning rates via param groups.

**Tech Stack:** `transformers` (`Trainer`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`),
`trl.chat_template_utils` (`add_response_schema`, `get_training_chat_template`, `parse_response`),
`torch`, `trackio`. Reuses this repo's existing `SearchEnv`, `rewards.py`'s `format_reward`/
`outcome_reward`, and `data.py`'s loaders unmodified.

## Global Constraints

These are fixed by the paper (Section 6.2/C.1.3) and the approved design spec
(`docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md`) — every task below must match them
exactly, not approximate them:

- `N_max = 4` (max turns per episode, this phase's own value — not the GRPO conditions'
  `max_tool_calling_iterations=4`, which is a different, unrelated cap).
- GAE: `gamma=1.0`, `gae_lambda=1.0`.
- PPO clip: `clip_eps=0.2`.
- KL penalty: `kl_beta=0.001`, evaluated against the rollout-time (frozen) policy snapshot, not a
  separate reference model.
- `policy_lr=1e-6`, `critic_lr=1e-5` (10x apart, per the paper).
- `num_ppo_epochs=4` inner update epochs per collected rollout batch.
- `value_loss_coef=0.5` — an assumed PPO-literature default, not paper-derived (documented as
  such in the design spec; do not present it as paper-grounded in code comments).
- Critic is a **separate model** (`AutoModelForSequenceClassification(num_labels=1)`), not a
  shared backbone with the policy.
- Model: `Qwen/Qwen3.5-0.8B` (same as the rest of this repo).
- `R^O` (`format_reward` + `outcome_reward`, summed) always lands on the trajectory's last
  action token. `R^I` (turn_reward's marginal per-turn contribution) lands at each intermediate
  turn boundary for `mt_ppo` only; always `0` for `ppo` (Eq. 9).
- Reuse `SearchEnv`/`rewards.py`/`data.py` **unmodified** except for the one named constant
  extraction in Task 1 below.
- `tests/unit/` only — no GPU, no live retrieval server, no network in unit tests (per CLAUDE.md's
  Guiding principles). The rollout loop, critic construction, and full trainer integration are
  **not** unit-tested — validated by the live smoke test (Task 11) instead, same principle as
  Phase 4's `build_trainer`.

---

## File Structure

- **Modify:** `src/turn_level_rewards/rewards.py` — extract the `0.4` turn-reward scale into a
  public, named constant so `train_ppo.py` can reuse the exact same value instead of duplicating
  the magic number (a real DRY risk: two independent `0.4` literals could silently drift apart).
- **Create:** `src/turn_level_rewards/train_ppo.py` — everything else: the two/three pure,
  unit-tested functions (`compute_gae`, `place_turn_rewards`, `compute_ppo_loss`), `MTPPOConfig`,
  `_PolicyAndCritic`, `build_policy_and_critic`, `MTPPOTrainer`, `build_ppo_config`,
  `build_ppo_trainer`, `main`.
- **Create:** `tests/unit/test_train_ppo.py` — tests for every pure function above plus
  `build_ppo_config`/`_parse_args`/`MTPPOTrainer.create_optimizer` (the last one is testable with
  tiny fake `nn.Linear` stand-ins for policy/critic, no real model needed).
- **Create:** `scripts/verify_phase7.py` — mirrors `scripts/verify_phase4.py`'s pattern exactly
  (run `tests/unit/`, `ruff check`, `ty check`, plus direct `build_ppo_config` assertions).

One file for the trainer (not split further) matches this repo's existing pattern — `train.py` is
also a single file containing its config-builder, callback, and composition root together.

---

### Task 1: Extract `TURN_REWARD_SCALE` as a public constant in `rewards.py`

**Files:**
- Modify: `src/turn_level_rewards/rewards.py`
- Test: `tests/unit/test_rewards.py`

**Interfaces:**
- Produces: `turn_level_rewards.rewards.TURN_REWARD_SCALE: float` (value `0.4`), importable by
  `train_ppo.py`'s `place_turn_rewards` in Task 3.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_rewards.py`:

```python
def test_turn_reward_scale_constant_matches_turn_reward_behavior():
    from turn_level_rewards.rewards import TURN_REWARD_SCALE

    assert TURN_REWARD_SCALE == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_rewards.py::test_turn_reward_scale_constant_matches_turn_reward_behavior -v`
Expected: FAIL with `ImportError: cannot import name 'TURN_REWARD_SCALE'`

- [ ] **Step 3: Write minimal implementation**

In `src/turn_level_rewards/rewards.py`, add a module-level constant near the top (after the
existing imports, before `_ANSWER_RE`) and use it in `turn_reward`:

```python
TURN_REWARD_SCALE = 0.4
```

Change `turn_reward`'s body from:

```python
        rewards.append(0.4 * environment.retrieval_fraction)
```

to:

```python
        rewards.append(TURN_REWARD_SCALE * environment.retrieval_fraction)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_rewards.py -v`
Expected: PASS (all existing `test_rewards.py` tests still pass too — this is a pure refactor, no
behavior change).

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/rewards.py tests/unit/test_rewards.py
git commit -m "rewards: extract TURN_REWARD_SCALE as a public constant

Needed by Phase 7's Eq. 9 reward-placement logic, which must use the exact
same 0.4 scale turn_reward already uses -- a named, shared constant avoids
two independent magic numbers silently drifting apart."
```

---

### Task 2: `compute_gae` pure function

**Files:**
- Create: `src/turn_level_rewards/train_ppo.py`
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Produces: `compute_gae(rewards: list[float], values: list[float], gamma: float = 1.0, lam: float = 1.0, bootstrap_value: float = 0.0) -> list[float]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_train_ppo.py`:

```python
"""Fast, GPU-free tests for train_ppo.py's pure functions and config builder.

No real MTPPOTrainer, model, or GPU is constructed here -- the rollout loop and critic
construction require a real model/chat-template, which is exactly what the live smoke test
(not tests/unit/) validates instead, per CLAUDE.md's Guiding principles.
"""

from turn_level_rewards.train_ppo import compute_gae


def test_compute_gae_matches_hand_computed_returns_minus_baseline_at_gamma_lambda_one():
    """At gamma=1, lambda=1 (this repo's fixed values), GAE reduces to
    (full-episode Monte-Carlo return from t) - V_t -- hand-computed here, not just re-deriving
    the recursive formula back at itself.

    rewards=[1.0, 0.0, 2.0], values=[0.5, 0.5, 0.5], bootstrap_value=0.0:
      return_2 = 2.0 + 0.0        = 2.0  -> A_2 = 2.0 - 0.5 = 1.5
      return_1 = 0.0 + return_2   = 2.0  -> A_1 = 2.0 - 0.5 = 1.5
      return_0 = 1.0 + return_1   = 3.0  -> A_0 = 3.0 - 0.5 = 2.5
    """
    advantages = compute_gae(rewards=[1.0, 0.0, 2.0], values=[0.5, 0.5, 0.5])

    assert advantages == [2.5, 1.5, 1.5]


def test_compute_gae_single_step_episode():
    advantages = compute_gae(rewards=[1.5], values=[0.2])

    assert advantages == [1.3]


def test_compute_gae_nonzero_bootstrap_value_feeds_into_final_step():
    advantages = compute_gae(rewards=[1.0], values=[0.5], bootstrap_value=2.0)

    # delta = r + gamma*bootstrap - V = 1.0 + 2.0 - 0.5 = 2.5
    assert advantages == [2.5]


def test_compute_gae_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError, match="equal length"):
        compute_gae(rewards=[1.0, 2.0], values=[0.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.train_ppo'`

- [ ] **Step 3: Write minimal implementation**

Create `src/turn_level_rewards/train_ppo.py`:

```python
"""train_ppo.py: custom multi-turn PPO trainer (MTPPOTrainer) for ppo/mt_ppo conditions.

Built directly on transformers.Trainer, not GRPOTrainer/PPOTrainer -- TRL's PPOTrainer has no
multi-turn tool-calling support (confirmed fresh against the installed 1.7.1 and upstream's
dev branch, re-verified 2026-07-23; see
docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md). Reuses SearchEnv/rewards.py/data.py
unmodified. See CLAUDE.md's Goal section and docs/phase-7-mt-ppo.md for the full design.
"""

from typing import Literal

Condition = Literal["ppo", "mt_ppo"]


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 1.0,
    lam: float = 1.0,
    bootstrap_value: float = 0.0,
) -> list[float]:
    """Generalized Advantage Estimation (standard recursive formula).

    len(values) must equal len(rewards) -- values[t] is the critic's estimate at position t.
    bootstrap_value is V for the (terminal) state after the last reward -- 0.0 for an episode
    that truly ends, since there's no further return to bootstrap from. At this repo's fixed
    gamma=1, lambda=1 (paper's own spec), this reduces toward a full-episode
    Monte-Carlo-return-minus-baseline -- no discount/decay tuning needed.
    """
    if len(rewards) != len(values):
        raise ValueError(
            f"rewards ({len(rewards)}) and values ({len(values)}) must be equal length"
        )
    advantages = [0.0] * len(rewards)
    running_gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running_gae = delta + gamma * lam * running_gae
        advantages[t] = running_gae
        next_value = values[t]
    return advantages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add compute_gae pure function"
```

---

### Task 3: `place_turn_rewards` pure function (Eq. 9)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Consumes: `turn_level_rewards.rewards.TURN_REWARD_SCALE` (Task 1).
- Produces: `place_turn_rewards(num_tokens: int, turn_boundary_token_indices: list[int], retrieval_fraction_after_each_turn: list[float], format_and_outcome_reward: float, condition: Condition, turn_reward_scale: float = TURN_REWARD_SCALE) -> list[float]`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_train_ppo.py`:

```python
from turn_level_rewards.train_ppo import place_turn_rewards


def test_place_turn_rewards_ppo_condition_never_places_turn_reward():
    """ppo: R^I is always 0 -- single lump-sum credit assignment even across a multi-turn
    episode. Only R^O (format_and_outcome_reward) lands, at the last token.
    """
    rewards = place_turn_rewards(
        num_tokens=10,
        turn_boundary_token_indices=[2, 5],
        retrieval_fraction_after_each_turn=[0.5, 1.0],
        format_and_outcome_reward=1.2,
        condition="ppo",
    )

    assert rewards == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2]


def test_place_turn_rewards_mt_ppo_places_marginal_retrieval_gain_at_each_turn_boundary():
    """mt_ppo: R^I at each intermediate turn boundary is the MARGINAL gain in retrieval_fraction
    that specific turn caused (0.5 at turn 1, then 1.0-0.5=0.5 at turn 2) -- not the raw
    cumulative value, which would double-count every later turn's contribution.
    """
    rewards = place_turn_rewards(
        num_tokens=10,
        turn_boundary_token_indices=[2, 5],
        retrieval_fraction_after_each_turn=[0.5, 1.0],
        format_and_outcome_reward=1.2,
        condition="mt_ppo",
        turn_reward_scale=0.4,
    )

    assert rewards[2] == 0.4 * 0.5
    assert rewards[5] == 0.4 * (1.0 - 0.5)
    assert rewards[-1] == 1.2
    assert rewards[0] == 0.0
    assert rewards[1] == 0.0
    assert rewards[3] == 0.0
    assert rewards[4] == 0.0


def test_place_turn_rewards_mt_ppo_with_zero_intermediate_turns_matches_ppo():
    """An episode that answers without ever calling search (no intermediate turns) should score
    identically in both conditions -- there's nothing for the turn_reward term to differentiate.
    """
    ppo_rewards = place_turn_rewards(
        num_tokens=4,
        turn_boundary_token_indices=[],
        retrieval_fraction_after_each_turn=[],
        format_and_outcome_reward=0.9,
        condition="ppo",
    )
    mt_ppo_rewards = place_turn_rewards(
        num_tokens=4,
        turn_boundary_token_indices=[],
        retrieval_fraction_after_each_turn=[],
        format_and_outcome_reward=0.9,
        condition="mt_ppo",
    )

    assert ppo_rewards == mt_ppo_rewards == [0.0, 0.0, 0.0, 0.9]


def test_place_turn_rewards_defaults_turn_reward_scale_to_the_shared_constant():
    from turn_level_rewards.rewards import TURN_REWARD_SCALE

    rewards = place_turn_rewards(
        num_tokens=3,
        turn_boundary_token_indices=[0],
        retrieval_fraction_after_each_turn=[1.0],
        format_and_outcome_reward=0.0,
        condition="mt_ppo",
    )

    assert rewards[0] == TURN_REWARD_SCALE


def test_place_turn_rewards_rejects_mismatched_boundary_and_fraction_lengths():
    import pytest

    with pytest.raises(ValueError, match="equal length"):
        place_turn_rewards(
            num_tokens=5,
            turn_boundary_token_indices=[1, 2],
            retrieval_fraction_after_each_turn=[0.5],
            format_and_outcome_reward=0.0,
            condition="mt_ppo",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -v -k place_turn_rewards`
Expected: FAIL with `ImportError: cannot import name 'place_turn_rewards'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/turn_level_rewards/train_ppo.py` (after `compute_gae`, and add the import at the top
of the file):

```python
from turn_level_rewards.rewards import TURN_REWARD_SCALE


def place_turn_rewards(
    num_tokens: int,
    turn_boundary_token_indices: list[int],
    retrieval_fraction_after_each_turn: list[float],
    format_and_outcome_reward: float,
    condition: Condition,
    turn_reward_scale: float = TURN_REWARD_SCALE,
) -> list[float]:
    """Eq. 9 turn-boundary reward placement.

    R^O (format_reward + outcome_reward, summed by the caller) always lands on the trajectory's
    last token. R^I -- turn_reward's marginal per-turn contribution -- lands at each intermediate
    turn boundary, mt_ppo only; always 0 for ppo (single lump-sum credit assignment even across a
    multi-turn episode, per the paper's Eq. 9).

    turn_boundary_token_indices and retrieval_fraction_after_each_turn operate over whatever
    token-index space the caller is using (this repo's MTPPOTrainer uses action-token-relative
    indices, i.e. only counting policy-generated tokens -- see _rollout_episode's docstring).
    retrieval_fraction_after_each_turn[i] is SearchEnv.retrieval_fraction sampled immediately
    after intermediate turn i's tool call executed. retrieval_fraction is monotonically
    non-decreasing (SearchEnv only ever adds to its hit set), so each turn's real, marginal
    contribution is that turn's value minus the previous turn's (0.0 before the first turn) --
    not the raw cumulative value, which would double-count every later turn's reward.
    """
    if len(turn_boundary_token_indices) != len(retrieval_fraction_after_each_turn):
        raise ValueError(
            "turn_boundary_token_indices and retrieval_fraction_after_each_turn must be equal "
            "length"
        )
    per_token_rewards = [0.0] * num_tokens
    per_token_rewards[-1] += format_and_outcome_reward
    if condition == "mt_ppo":
        previous_fraction = 0.0
        for token_index, cumulative_fraction in zip(
            turn_boundary_token_indices, retrieval_fraction_after_each_turn, strict=True
        ):
            marginal = cumulative_fraction - previous_fraction
            per_token_rewards[token_index] += turn_reward_scale * marginal
            previous_fraction = cumulative_fraction
    return per_token_rewards
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add place_turn_rewards pure function (Eq. 9)"
```

---

### Task 4: `compute_ppo_loss` pure function

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Produces: `compute_ppo_loss(new_logprobs, old_logprobs, advantages, returns, new_values, action_mask, clip_eps=0.2, kl_beta=0.001, value_loss_coef=0.5) -> dict[str, torch.Tensor]`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_train_ppo.py`:

```python
import torch

from turn_level_rewards.train_ppo import compute_ppo_loss


def test_compute_ppo_loss_zero_advantage_and_matched_values_gives_only_kl_term():
    """advantages=0 -> policy_loss term is 0 regardless of ratio; new_values==returns ->
    value_loss is 0; new_logprobs==old_logprobs -> ratio==1 and kl==0. Total loss should be
    exactly 0.0 in this fully-matched case.
    """
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        old_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        advantages=torch.tensor([0.0, 0.0, 0.0]),
        returns=torch.tensor([1.0, 1.0, 1.0]),
        new_values=torch.tensor([1.0, 1.0, 1.0]),
        action_mask=torch.tensor([1.0, 1.0, 1.0]),
    )

    assert result["loss"].item() == 0.0
    assert result["policy_loss"].item() == 0.0
    assert result["value_loss"].item() == 0.0
    assert result["kl"].item() == 0.0


def test_compute_ppo_loss_clips_large_positive_ratio_on_positive_advantage():
    """A large ratio (new much more likely than old) on positive advantage should be clipped to
    (1+clip_eps), not allowed to blow up the policy objective unbounded.
    """
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([10.0]),  # ratio = exp(10) >> 1 + clip_eps
        old_logprobs=torch.tensor([0.0]),
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([0.0]),
        new_values=torch.tensor([0.0]),
        action_mask=torch.tensor([1.0]),
        clip_eps=0.2,
        kl_beta=0.0,
        value_loss_coef=0.0,
    )

    # unclipped would be -(exp(10) * 1.0); clipped surrogate must use min(unclipped, clipped) --
    # since advantage is positive, clipping caps the objective at 1.2 * 1.0, so policy_loss is
    # exactly -1.2, not some huge negative number.
    assert result["policy_loss"].item() == pytest.approx(-1.2, abs=1e-4)


def test_compute_ppo_loss_masks_out_non_action_positions():
    """A masked-out position (action_mask=0) with wildly wrong values must not affect the loss at
    all -- only masked-in (action_mask=1) positions should contribute.
    """
    masked_result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0, 999.0]),
        old_logprobs=torch.tensor([0.0, -999.0]),
        advantages=torch.tensor([0.0, 999.0]),
        returns=torch.tensor([1.0, -999.0]),
        new_values=torch.tensor([1.0, 999.0]),
        action_mask=torch.tensor([1.0, 0.0]),
    )
    unmasked_result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0]),
        old_logprobs=torch.tensor([0.0]),
        advantages=torch.tensor([0.0]),
        returns=torch.tensor([1.0]),
        new_values=torch.tensor([1.0]),
        action_mask=torch.tensor([1.0]),
    )

    assert masked_result["loss"].item() == pytest.approx(unmasked_result["loss"].item())


def test_compute_ppo_loss_value_loss_scales_with_squared_error():
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0]),
        old_logprobs=torch.tensor([0.0]),
        advantages=torch.tensor([0.0]),
        returns=torch.tensor([3.0]),
        new_values=torch.tensor([1.0]),
        action_mask=torch.tensor([1.0]),
        kl_beta=0.0,
    )

    assert result["value_loss"].item() == pytest.approx(4.0)  # (1.0 - 3.0)**2
    assert result["loss"].item() == pytest.approx(0.5 * 4.0)  # value_loss_coef defaults to 0.5
```

Add `import pytest` to the top of `tests/unit/test_train_ppo.py` if not already present (it is,
from Task 2's `ValueError` tests using inline imports — move `import pytest` to the top-level
import block instead, alongside `import torch`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -v -k compute_ppo_loss`
Expected: FAIL with `ImportError: cannot import name 'compute_ppo_loss'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/turn_level_rewards/train_ppo.py` (add `import torch` at the top of the file):

```python
import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.001,
    value_loss_coef: float = 0.5,
) -> dict[str, torch.Tensor]:
    """PPO-clip policy loss + value loss + a direct KL penalty term, all masked to action tokens.

    action_mask is 1.0 at positions that are real policy-generated action tokens, 0.0 elsewhere
    (prompt tokens, tool-response tokens injected by the environment, padding) -- none of those
    should ever receive policy gradient. The KL term uses old_logprobs (the rollout-time, frozen
    policy snapshot) as the reference throughout every one of this batch's inner PPO epochs --
    not a separate frozen reference model the way GRPO's beta works (see the design spec's
    stated assumption on this point). Both the clip and the KL term are applied together, not
    either/or, per the paper's spec.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -_masked_mean(torch.min(unclipped, clipped), action_mask)
    value_loss = _masked_mean((new_values - returns) ** 2, action_mask)
    kl = _masked_mean(new_logprobs - old_logprobs, action_mask)
    loss = policy_loss + value_loss_coef * value_loss + kl_beta * kl
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.detach(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: PASS (13 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add compute_ppo_loss pure function (clip + KL + value loss)"
```

---

### Task 5: `MTPPOConfig` + `build_ppo_config`

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Produces: `MTPPOConfig` (a `TrainingArguments` subclass), `build_ppo_config(condition: Condition, seed: int, max_steps: int, num_rollouts_per_step: int) -> MTPPOConfig`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_train_ppo.py`:

```python
from turn_level_rewards.train_ppo import build_ppo_config


def test_build_ppo_config_fixed_hyperparameters_identical_across_conditions():
    """These come from the paper (Section 6.2/C.1.3) or the design spec's stated assumptions --
    every one must hold for BOTH conditions, since ppo/mt_ppo differ only in reward placement
    (Eq. 9), not in any of these hyperparameters.
    """
    ppo_config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    mt_ppo_config = build_ppo_config("mt_ppo", seed=42, max_steps=2, num_rollouts_per_step=2)

    for config in (ppo_config, mt_ppo_config):
        assert config.n_max == 4
        assert config.clip_eps == 0.2
        assert config.kl_beta == 0.001
        assert config.policy_lr == 1e-6
        assert config.critic_lr == 1e-5
        assert config.gamma == 1.0
        assert config.gae_lambda == 1.0
        assert config.num_ppo_epochs == 4
        assert config.value_loss_coef == 0.5
        assert config.max_completion_length == 2048


def test_build_ppo_config_condition_and_derived_fields_differ():
    ppo_config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    mt_ppo_config = build_ppo_config("mt_ppo", seed=42, max_steps=2, num_rollouts_per_step=2)

    assert ppo_config.condition == "ppo"
    assert mt_ppo_config.condition == "mt_ppo"
    assert ppo_config.output_dir == "outputs/ppo"
    assert mt_ppo_config.output_dir == "outputs/mt_ppo"
    assert ppo_config.run_name == "ppo"
    assert mt_ppo_config.run_name == "mt_ppo"


def test_build_ppo_config_passes_through_seed_max_steps_and_rollout_count():
    config = build_ppo_config("ppo", seed=7, max_steps=500, num_rollouts_per_step=8)

    assert config.seed == 7
    assert config.max_steps == 500
    assert config.num_rollouts_per_step == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -v -k build_ppo_config`
Expected: FAIL with `ImportError: cannot import name 'build_ppo_config'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/turn_level_rewards/train_ppo.py` (add `from dataclasses import dataclass` and
`from transformers import TrainingArguments` to the top of the file):

```python
from dataclasses import dataclass

from transformers import TrainingArguments

MODEL_NAME = "Qwen/Qwen3.5-0.8B"


@dataclass
class MTPPOConfig(TrainingArguments):
    """Config for MTPPOTrainer. Subclasses TrainingArguments the same way TRL's GRPOConfig does,
    adding this trainer's own fixed hyperparameters (see this plan's Global Constraints section
    for where each value comes from).
    """

    condition: Condition = "ppo"
    n_max: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.001
    policy_lr: float = 1e-6
    critic_lr: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    num_ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    num_rollouts_per_step: int = 2
    max_completion_length: int = 2048
    project: str = "turn-level-rewards-ppo"


def build_ppo_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_rollouts_per_step: int,
) -> MTPPOConfig:
    """Build the MTPPOConfig for a training run. Mirrors train.py's build_config role from
    Phase 4 -- fixed hyperparameters are baked in here, not exposed as independent CLI flags.
    """
    return MTPPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        condition=condition,
        n_max=4,
        clip_eps=0.2,
        kl_beta=0.001,
        policy_lr=1e-6,
        critic_lr=1e-5,
        gamma=1.0,
        gae_lambda=1.0,
        num_ppo_epochs=4,
        value_loss_coef=0.5,
        num_rollouts_per_step=num_rollouts_per_step,
        max_completion_length=2048,
        logging_steps=1,
        run_name=condition,
        report_to="none",  # trackio is called directly in MTPPOTrainer.train(), not through
                            # transformers' generic report_to integration -- see Task 8.
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: PASS (16 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add MTPPOConfig + build_ppo_config"
```

---

### Task 6: `_PolicyAndCritic` + `build_policy_and_critic` + `create_optimizer` (tested with fake models)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Consumes: `MTPPOConfig` (Task 5).
- Produces: `_PolicyAndCritic(nn.Module)`, `build_policy_and_critic(model_name: str = MODEL_NAME) -> _PolicyAndCritic`, `MTPPOTrainer.create_optimizer(self) -> torch.optim.Optimizer` (the first method on `MTPPOTrainer`, introduced here since it's unit-testable without a real model).

This task introduces the `MTPPOTrainer` class itself (minimal `__init__` + `create_optimizer`
only) — later tasks add its rollout/update methods.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_train_ppo.py`:

```python
import torch.nn as nn

from turn_level_rewards.train_ppo import MTPPOTrainer, _PolicyAndCritic


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _FakeCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)


def test_create_optimizer_uses_two_param_groups_with_paper_learning_rates(tmp_path):
    config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    config.output_dir = str(tmp_path)
    model = _PolicyAndCritic(_FakePolicy(), _FakeCritic())
    trainer = MTPPOTrainer.__new__(MTPPOTrainer)  # bypass __init__ (needs a real tokenizer)
    trainer.model = model
    trainer.args = config

    optimizer = trainer.create_optimizer()

    assert len(optimizer.param_groups) == 2
    policy_group, critic_group = optimizer.param_groups
    assert policy_group["lr"] == 1e-6
    assert critic_group["lr"] == 1e-5
    assert list(policy_group["params"]) == list(model.policy.parameters())
    assert list(critic_group["params"]) == list(model.critic.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -v -k create_optimizer`
Expected: FAIL with `ImportError: cannot import name 'MTPPOTrainer'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/turn_level_rewards/train_ppo.py` (add these imports at the top:
`import torch.nn as nn`, `from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, PreTrainedModel, Trainer`):

```python
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    Trainer,
)
from trl.chat_template_utils import add_response_schema, get_training_chat_template

from turn_level_rewards.env import SearchEnv


class _PolicyAndCritic(nn.Module):
    """Wraps the policy (causal LM) and critic (sequence-classification head) as one nn.Module.

    Lets Trainer's standard model/optimizer/checkpoint plumbing see both submodules through a
    single self.model, while create_optimizer still gives them independent learning rates (the
    paper's own spec: policy_lr=1e-6, critic_lr=1e-5) via separate param groups.
    """

    def __init__(self, policy: PreTrainedModel, critic: PreTrainedModel) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic


def build_policy_and_critic(model_name: str = MODEL_NAME) -> _PolicyAndCritic:
    """Real policy + real critic, both loaded from the same base checkpoint -- separate models,
    not a shared backbone (the paper's own spec). Not unit-tested: loads real weights: validated
    by the live smoke test (Task 11) instead.
    """
    policy = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    critic = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, dtype=torch.bfloat16
    )
    return _PolicyAndCritic(policy, critic)


class MTPPOTrainer(Trainer):
    """Custom multi-turn PPO trainer with tool-calling, built directly on transformers.Trainer.

    Owns: the rollout loop (render with tools -> generate -> parse_response -> execute
    SearchEnv.search() on a tool call -> append tool message -> repeat up to args.n_max turns ->
    require a final <answer>), the critic forward pass, Eq. 9 reward placement, GAE, and the
    PPO-clip + KL-penalty + value-loss update. Turn-level credit assignment for mt_ppo falls out
    of reward placement + GAE bootstrapping alone -- no MT-GRPO-style extra-rollout advantage
    trick is needed here (PPO already has a real per-token critic). See
    docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md's Context section for why this is
    built on transformers.Trainer directly rather than subclassing GRPOTrainer/PPOTrainer.
    """

    def __init__(
        self,
        condition: Condition,
        model: _PolicyAndCritic,
        tokenizer,
        train_dataset,
        args: MTPPOConfig,
        environment_factory=None,
        callbacks=None,
    ) -> None:
        self.condition = condition
        self.environment_factory = environment_factory or SearchEnv
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=callbacks)
        self.tokenizer = add_response_schema(tokenizer)
        self.training_chat_template = get_training_chat_template(self.tokenizer)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """One AdamW, two param groups -- policy_lr / critic_lr per the paper's spec (10x apart)."""
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.policy.parameters(), "lr": self.args.policy_lr},
                {"params": self.model.critic.parameters(), "lr": self.args.critic_lr},
            ]
        )
        return self.optimizer
```

Note: the test uses `MTPPOTrainer.__new__(MTPPOTrainer)` to bypass `__init__` entirely (setting
`trainer.model`/`trainer.args` directly), since constructing a real tokenizer/chat template isn't
needed to test `create_optimizer` in isolation — that bypass works regardless of where imports
live, since module-level imports execute at import time either way.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add _PolicyAndCritic, build_policy_and_critic, MTPPOTrainer.create_optimizer"
```

---

### Task 7: `MTPPOTrainer._rollout_episode` (real model, not unit-tested)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`

**Interfaces:**
- Consumes: `MTPPOConfig.n_max`, `MTPPOConfig.max_completion_length`, `self.environment_factory`
  (Task 6), a dataset row shaped like `data.py`'s shared contract (`prompt`, `question`,
  `golden_answers`, `metadata`).
- Produces: `MTPPOTrainer._rollout_episode(self, row: dict) -> dict` with keys `row`, `completion`,
  `full_token_ids`, `action_mask`, `turn_boundary_action_indices`,
  `retrieval_fraction_after_each_turn` — consumed by Task 8's `_forward_logprobs_and_values` and
  `_collect_batch`.

This task also needs `parse_response` (Task 6 already imported `SearchEnv`, `add_response_schema`,
`get_training_chat_template` at the top of the file) — add it to that same import line:

```python
from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response
```

**Not unit-tested** — this method requires a real model, tokenizer, chat template, and (for the
live smoke test) a running retrieval server. This is exactly the integration surface Phase 4's
`build_trainer` was also not unit-tested for; validated instead by Task 11's live smoke test.

- [ ] **Step 1: Write the implementation**

Add to `src/turn_level_rewards/train_ppo.py`, inside `MTPPOTrainer`:

```python
    def _rollout_episode(self, row: dict) -> dict:
        """Run one multi-turn episode for a single dataset row: real generation, real tool calls
        against the real retrieval server (via self.environment_factory, e.g. SearchEnv).

        Returns everything downstream needs, all expressed over a "compressed action-token"
        index space that only counts policy-generated (assistant) tokens -- prompt tokens and
        tool-response tokens (environment-injected, not sampled by the policy) are excluded from
        this space entirely, matching how GAE/PPO treat each policy-generated token as one RL
        timestep:
          - full_token_ids: every token in the final rendered conversation (prompt + all turns),
            in order -- fed to the policy/critic for full context.
          - action_mask: same length as full_token_ids, 1 at positions the policy generated,
            0 elsewhere.
          - turn_boundary_action_indices: for each intermediate turn (one that made a tool call,
            i.e. not the final answering turn), the index INTO THE COMPRESSED ACTION-TOKEN
            SEQUENCE (not full_token_ids) of that turn's last generated token.
          - retrieval_fraction_after_each_turn: SearchEnv.retrieval_fraction sampled immediately
            after that same turn's tool call executed -- one entry per turn_boundary_action_index,
            same order.

        Relies on get_training_chat_template's prefix-preserving guarantee (confirmed supported
        for Qwen3.5): each turn's freshly-rendered prompt is guaranteed to start with exactly the
        tokens already recorded in full_token_ids, so the new suffix at each turn is unambiguous.
        """
        environment = self.environment_factory()
        environment.reset(**row)
        messages = list(row["prompt"])
        policy = self.model.policy

        full_token_ids: list[int] = []
        action_mask: list[int] = []
        turn_boundary_action_indices: list[int] = []
        retrieval_fraction_after_each_turn: list[float] = []
        num_action_tokens = 0

        for _turn in range(self.args.n_max):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tools=[environment.search],
                add_generation_prompt=True,
                chat_template=self.training_chat_template,
                tokenize=False,
            )
            prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            new_context_tokens = prompt_token_ids[len(full_token_ids) :]
            full_token_ids.extend(new_context_tokens)
            action_mask.extend([0] * len(new_context_tokens))

            input_ids = torch.tensor([prompt_token_ids], device=policy.device)
            with torch.no_grad():
                generation = policy.generate(
                    input_ids,
                    max_new_tokens=self.args.max_completion_length,
                    do_sample=True,
                    temperature=1.0,
                )
            new_token_ids = generation[0, len(prompt_token_ids) :].tolist()
            parsed = parse_response(self.tokenizer, new_token_ids, prefix=prompt_token_ids)
            messages.append(parsed)

            full_token_ids.extend(new_token_ids)
            action_mask.extend([1] * len(new_token_ids))
            num_action_tokens += len(new_token_ids)

            tool_calls = parsed.get("tool_calls") or []
            if not tool_calls:
                break  # final answer turn -- episode complete

            for tool_call in tool_calls:
                result = environment.search(**tool_call["function"]["arguments"])
                messages.append({"role": "tool", "name": "search", "content": result})

            turn_boundary_action_indices.append(num_action_tokens - 1)
            retrieval_fraction_after_each_turn.append(environment.retrieval_fraction)

        completion = messages[len(row["prompt"]) :]
        return {
            "row": row,
            "completion": completion,
            "full_token_ids": full_token_ids,
            "action_mask": action_mask,
            "turn_boundary_action_indices": turn_boundary_action_indices,
            "retrieval_fraction_after_each_turn": retrieval_fraction_after_each_turn,
        }
```

- [ ] **Step 2: No automated test for this step** (real model/GPU/retrieval-server required — see
      Task 11's live smoke test).

- [ ] **Step 3: Run the existing unit test suite to confirm no regression**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS (17 tests total, unchanged from Task 6 — this task adds no new unit-tested surface)

- [ ] **Step 4: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py
git commit -m "train_ppo: add MTPPOTrainer._rollout_episode (real multi-turn tool-calling loop)"
```

---

### Task 8: `_forward_logprobs_and_values` (real model, not unit-tested)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`

**Interfaces:**
- Consumes: `rollout["full_token_ids"]`, `rollout["action_mask"]` (Task 7).
- Produces: `MTPPOTrainer._forward_logprobs_and_values(self, full_token_ids: list[int], action_mask: list[int]) -> tuple[torch.Tensor, torch.Tensor]` — `(action_logprobs, action_values)`, both 1-D tensors of length `sum(action_mask)`, consumed by Task 9's `_collect_batch` and `_ppo_update`.

**Not unit-tested** — requires the real policy/critic forward pass.

- [ ] **Step 1: Write the implementation**

Add to `src/turn_level_rewards/train_ppo.py`, inside `MTPPOTrainer`:

```python
    def _forward_logprobs_and_values(
        self, full_token_ids: list[int], action_mask: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass over one episode's full token sequence.

        Returns (action_logprobs, action_values), both 1-D tensors of length sum(action_mask) --
        one entry per action (policy-generated) token, in the same order as
        _rollout_episode's turn_boundary_action_indices.

        Called twice per episode per PPO update: once under torch.no_grad() right after rollout
        (the frozen "old" reference for the clip ratio, GAE, and the KL term), and once per PPO
        inner epoch WITH grad (the "new" values that get backpropagated).
        """
        device = self.model.policy.device
        input_ids = torch.tensor([full_token_ids], device=device)
        mask = torch.tensor(action_mask, device=device, dtype=torch.bool)

        policy_logits = self.model.policy(input_ids=input_ids).logits[0]  # [seq_len, vocab]
        # logits[t] predicts token[t+1]; gather log-prob of the actual next token at each
        # position, then select the ones landing on action tokens (shifted by one: an action
        # token at absolute position t was predicted by logits at position t-1).
        log_probs_all = torch.log_softmax(policy_logits[:-1], dim=-1)
        next_tokens = input_ids[0, 1:]
        token_logprobs = log_probs_all.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
        action_indices = mask.nonzero(as_tuple=True)[0]
        action_logprobs = token_logprobs[action_indices - 1]

        critic_hidden = self.model.critic.model(input_ids=input_ids).last_hidden_state
        critic_values = self.model.critic.score(critic_hidden).squeeze(-1)[0]  # [seq_len]
        action_values = critic_values[action_indices]

        return action_logprobs, action_values
```

- [ ] **Step 2: No automated test for this step** (real model/GPU required).

- [ ] **Step 3: Run the existing unit test suite to confirm no regression**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS (17 tests total)

- [ ] **Step 4: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py
git commit -m "train_ppo: add MTPPOTrainer._forward_logprobs_and_values"
```

---

> **Amendment (added after Task 7 landed, during execution):** the user asked for detailed
> diagnostic logging (so a failed or surprising run can be diagnosed without rerunning it),
> deterministic repeatability, and progress visibility on long runs. Tasks 9, 10, and 13 below
> were revised to add this before they were implemented — `_collect_batch` now carries
> `question`/`completion` through to `train()`, `train()` calls `set_seed` and writes two
> diagnostic artifacts (`train_log.jsonl`, `sample_completions.log`) plus a stdout progress line,
> and Task 13's smoke test verifies both artifacts are actually populated with real content.

### Task 9: `_collect_batch` (real model, not unit-tested)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`

**Interfaces:**
- Consumes: `_rollout_episode` (Task 7), `_forward_logprobs_and_values` (Task 8),
  `place_turn_rewards` (Task 3), `compute_gae` (Task 2), `rewards.format_reward`/
  `rewards.outcome_reward`.
- Produces: `MTPPOTrainer._collect_batch(self, rows: list[dict]) -> list[dict]` — one dict per
  episode with keys `full_token_ids`, `action_mask`, `old_logprobs`, `old_values`, `advantages`,
  `returns`, `format_and_outcome_reward`, `retrieval_fraction`, plus `question` and `completion`
  (the row's question text and the episode's message list) — consumed by Task 10's
  `_ppo_update`/`train()`, where `question`/`completion` feed the per-step diagnostic log and the
  periodic sample-completion log (added in response to a mid-plan request for detailed,
  rerun-free diagnostics on long training runs — see this plan's amendment note above Task 9).

Add the needed import at the top of the file: `from turn_level_rewards.rewards import format_reward, outcome_reward`
(alongside the existing `TURN_REWARD_SCALE` import from Task 3 — combine into one import line).

- [ ] **Step 1: Write the implementation**

Add to `src/turn_level_rewards/train_ppo.py`, inside `MTPPOTrainer`:

```python
    def _collect_batch(self, rows: list[dict]) -> list[dict]:
        """Roll out one episode per row, score it, and compute its frozen GAE inputs.

        Not unit-tested -- calls _rollout_episode (real model/tool-calls) and
        _forward_logprobs_and_values (real forward pass) for each row. Validated by the live
        smoke test (Task 11).
        """
        episodes = []
        for row in rows:
            rollout = self._rollout_episode(row)

            with torch.no_grad():
                old_logprobs, old_values = self._forward_logprobs_and_values(
                    rollout["full_token_ids"], rollout["action_mask"]
                )

            completion = rollout["completion"]
            format_r = format_reward([completion])[0]
            outcome_r = outcome_reward([completion], [row["golden_answers"]])[0]
            format_and_outcome_reward = format_r + outcome_r

            retrieval_fraction = (
                rollout["retrieval_fraction_after_each_turn"][-1]
                if rollout["retrieval_fraction_after_each_turn"]
                else 0.0
            )

            per_token_rewards = place_turn_rewards(
                num_tokens=len(old_values),
                turn_boundary_token_indices=rollout["turn_boundary_action_indices"],
                retrieval_fraction_after_each_turn=rollout["retrieval_fraction_after_each_turn"],
                format_and_outcome_reward=format_and_outcome_reward,
                condition=self.condition,
            )
            advantages = compute_gae(
                rewards=per_token_rewards,
                values=old_values.tolist(),
                gamma=self.args.gamma,
                lam=self.args.gae_lambda,
            )
            returns = [a + v for a, v in zip(advantages, old_values.tolist(), strict=True)]

            episodes.append(
                {
                    "full_token_ids": rollout["full_token_ids"],
                    "action_mask": rollout["action_mask"],
                    "old_logprobs": old_logprobs,
                    "old_values": old_values,
                    "advantages": torch.tensor(advantages, device=old_values.device),
                    "returns": torch.tensor(returns, device=old_values.device),
                    "format_and_outcome_reward": format_and_outcome_reward,
                    "retrieval_fraction": retrieval_fraction,
                    "question": row["question"],
                    "completion": completion,
                }
            )
        return episodes
```

`question`/`completion` are not needed by GAE/PPO math -- they exist purely so `train()` (Task
10) can write diagnostic logs (a per-step JSONL record and a periodic human-readable sample
completion) without needing to re-derive them or re-run anything.

- [ ] **Step 2: No automated test for this step** (real model required).

- [ ] **Step 3: Run the existing unit test suite to confirm no regression**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS (17 tests total)

- [ ] **Step 4: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py
git commit -m "train_ppo: add MTPPOTrainer._collect_batch"
```

---

### Task 10: `_ppo_update` + `train()` override (real model, not unit-tested)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`

**Interfaces:**
- Consumes: `_collect_batch` (Task 9), `compute_ppo_loss` (Task 4), `create_optimizer` (Task 6).
- Produces: `MTPPOTrainer._ppo_update(self, episodes: list[dict]) -> dict[str, float]` (mean
  loss/policy_loss/value_loss/kl across the inner epoch), `MTPPOTrainer.train(self) -> None`
  (overrides `Trainer.train()` entirely with the outer rollout-collection loop) — the last piece
  needed for `build_ppo_trainer` (Task 11) to produce a runnable trainer. `train()` also writes
  two diagnostic artifacts under `self.args.output_dir` — `train_log.jsonl` (one structured JSON
  line per step) and `sample_completions.log` (periodic human-readable transcripts) — plus a
  stdout progress line per step, per this plan's mid-execution amendment (see the note above
  Task 9).

Add these imports at the top of the file: `import itertools`, `import json`, `import time`,
`from pathlib import Path`, `import trackio`, `from transformers.trainer_utils import set_seed`.

- [ ] **Step 1: Write the implementation**

Add to `src/turn_level_rewards/train_ppo.py`, inside `MTPPOTrainer`:

```python
    def _ppo_update(self, episodes: list[dict]) -> dict[str, float]:
        """Run args.num_ppo_epochs inner passes over the collected batch, gradient-accumulated
        across all episodes in the batch, one optimizer step per inner epoch. Matches this
        repo's existing train.py precedent of per-episode (batch-of-1) forward/backward passes
        with gradient accumulation, avoiding any padding/attention-mask complexity -- consistent
        with the single-RTX-4090, 0.8B-model memory profile the rest of this repo already
        established.
        """
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
        num_updates = 0
        for _epoch in range(self.args.num_ppo_epochs):
            self.optimizer.zero_grad()
            for episode in episodes:
                new_logprobs, new_values = self._forward_logprobs_and_values(
                    episode["full_token_ids"], episode["action_mask"]
                )
                action_mask = torch.ones_like(new_logprobs)
                loss_dict = compute_ppo_loss(
                    new_logprobs=new_logprobs,
                    old_logprobs=episode["old_logprobs"],
                    advantages=episode["advantages"],
                    returns=episode["returns"],
                    new_values=new_values,
                    action_mask=action_mask,
                    clip_eps=self.args.clip_eps,
                    kl_beta=self.args.kl_beta,
                    value_loss_coef=self.args.value_loss_coef,
                )
                (loss_dict["loss"] / len(episodes)).backward()
                for key in totals:
                    totals[key] += loss_dict[key].item()
                num_updates += 1
            self.optimizer.step()
        return {key: value / num_updates for key, value in totals.items()}
```

Add this constant near the top of the file, alongside `MODEL_NAME` (module level, not inside the
class):

```python
_SAMPLE_COMPLETION_INTERVAL = 10
```

Every 10th step, plus always step 0 (so even a 2-step smoke test produces at least one sample),
`train()` appends one full episode transcript to `sample_completions.log`. Frequent enough to
catch a policy that's degenerated mid-run without waiting for the run to finish; infrequent
enough not to flood the file on a long run.

```python
    def train(self) -> None:
        """Overrides Trainer.train() entirely: PPO's collect-then-multi-epoch-update structure
        doesn't fit Trainer's default single-pass-per-batch loop, so this owns the whole outer
        loop instead of relying on get_train_dataloader()/training_step().

        Writes two diagnostic artifacts under self.args.output_dir, so a run can be inspected
        after the fact without rerunning it:
          - train_log.jsonl: one JSON line per step with every metric plus a per-episode
            breakdown (question, reward, retrieval_fraction, action-token count) -- enough detail
            to diagnose a specific step or a specific episode's reward after training has already
            finished, not just an aggregate curve.
          - sample_completions.log: one full example transcript appended every
            _SAMPLE_COMPLETION_INTERVAL steps (plain text), mirroring this repo's existing
            train.py convention of log_completions=True for GRPO -- lets a human spot-check real
            model output during a long run without re-running anything.
        Also prints a one-line progress summary to stdout each step (step/max_steps, key metrics,
        elapsed and estimated-remaining wall-clock) so a long run's progress is visible without
        having trackio's dashboard open.

        set_seed(self.args.seed) is called here explicitly because this override replaces
        Trainer.train() entirely -- the base class's own seeding call never runs, so without this,
        two runs with the same --seed would silently stop reproducing the same rollouts (sampling
        in _rollout_episode uses torch's global RNG).
        """
        set_seed(self.args.seed)
        self.optimizer = self.create_optimizer()
        rows = list(self.train_dataset)
        row_cycle = itertools.cycle(rows)
        trackio.init(project=self.args.project, name=self.args.run_name)

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.jsonl"
        sample_completions_path = output_dir / "sample_completions.log"

        run_start = time.monotonic()
        for step in range(self.args.max_steps):
            step_start = time.monotonic()
            batch_rows = [next(row_cycle) for _ in range(self.args.num_rollouts_per_step)]
            episodes = self._collect_batch(batch_rows)
            update_metrics = self._ppo_update(episodes)

            mean_reward = sum(e["format_and_outcome_reward"] for e in episodes) / len(episodes)
            mean_retrieval_fraction = sum(e["retrieval_fraction"] for e in episodes) / len(
                episodes
            )
            total_elapsed = time.monotonic() - run_start
            steps_remaining = self.args.max_steps - (step + 1)
            eta_seconds = (total_elapsed / (step + 1)) * steps_remaining

            metrics = {
                "step": step,
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "kl": update_metrics["kl"],
                "reward": mean_reward,
                "retrieval_fraction": mean_retrieval_fraction,
            }
            trackio.log(metrics)

            log_record = dict(metrics)
            log_record["step_elapsed_seconds"] = time.monotonic() - step_start
            log_record["total_elapsed_seconds"] = total_elapsed
            log_record["eta_seconds"] = eta_seconds
            log_record["episodes"] = [
                {
                    "question": episode["question"],
                    "format_and_outcome_reward": episode["format_and_outcome_reward"],
                    "retrieval_fraction": episode["retrieval_fraction"],
                    "num_action_tokens": len(episode["old_values"]),
                }
                for episode in episodes
            ]
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(log_record) + "\n")

            if step == 0 or (step + 1) % _SAMPLE_COMPLETION_INTERVAL == 0:
                sample_episode = episodes[0]
                with sample_completions_path.open("a") as sample_file:
                    sample_file.write(
                        f"=== step {step + 1} | reward="
                        f"{sample_episode['format_and_outcome_reward']:.3f} | retrieval_fraction="
                        f"{sample_episode['retrieval_fraction']:.3f} ===\n"
                    )
                    sample_file.write(f"question: {sample_episode['question']}\n")
                    for message in sample_episode["completion"]:
                        sample_file.write(f"[{message.get('role')}] {message.get('content')}\n")
                    sample_file.write("\n")

            print(
                f"step {step + 1}/{self.args.max_steps} | loss={metrics['loss']:.4f} "
                f"reward={mean_reward:.3f} retrieval_fraction={mean_retrieval_fraction:.3f} "
                f"| elapsed={total_elapsed:.0f}s eta={eta_seconds:.0f}s",
                flush=True,
            )

            self.state.global_step = step + 1

            if (step + 1) % self.args.save_steps == 0 if self.args.save_steps else False:
                self.save_model(f"{self.args.output_dir}/checkpoint-{step + 1}")

        self.save_model(f"{self.args.output_dir}/checkpoint-{self.args.max_steps}")
```

- [ ] **Step 2: No automated test for this step** (real model required).

- [ ] **Step 3: Run the existing unit test suite to confirm no regression**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS (17 tests total)

- [ ] **Step 4: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py
git commit -m "train_ppo: add MTPPOTrainer._ppo_update and train() override with diagnostic logging"
```

---

### Task 11: `build_ppo_trainer` + `main()` CLI entrypoint

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py`
- Test: `tests/unit/test_train_ppo.py` (for `_parse_args` only — `build_ppo_trainer`/`main` are not
  unit-tested, same as `train.py`'s `build_trainer`/`main`).

**Interfaces:**
- Consumes: `build_ppo_config` (Task 5), `build_policy_and_critic` (Task 6), `MTPPOTrainer` (Tasks
  6-10), `data.load_train_dataset` (existing, unmodified).
- Produces: `_parse_args(argv: list[str] | None = None) -> argparse.Namespace`,
  `build_ppo_trainer(condition: Condition, train_size: int | None, config: MTPPOConfig) -> MTPPOTrainer`,
  `main() -> None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_train_ppo.py`:

```python
from turn_level_rewards.train_ppo import _parse_args


def test_parse_args_defaults():
    args = _parse_args(["--condition", "ppo"])

    assert args.condition == "ppo"
    assert args.seed == 42
    assert args.train_size == 8
    assert args.max_steps == 2
    assert args.num_rollouts_per_step == 2


def test_parse_args_condition_required():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_condition_choices_enforced():
    with pytest.raises(SystemExit):
        _parse_args(["--condition", "not_a_real_condition"])


def test_parse_args_overrides():
    args = _parse_args(
        [
            "--condition",
            "mt_ppo",
            "--seed",
            "7",
            "--train-size",
            "90447",
            "--max-steps",
            "500",
            "--num-rollouts-per-step",
            "8",
        ]
    )

    assert args.condition == "mt_ppo"
    assert args.seed == 7
    assert args.train_size == 90447
    assert args.max_steps == 500
    assert args.num_rollouts_per_step == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -v -k parse_args`
Expected: FAIL with `ImportError: cannot import name '_parse_args'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/turn_level_rewards/train_ppo.py` (add `import argparse` at the top):

```python
import argparse


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse train_ppo.py's CLI arguments. Mirrors train.py's _parse_args pattern from Phase 4 --
    the bare invocation (just --condition) is a tiny smoke-test-scale run; full runs (Phase 7b)
    must explicitly override --train-size/--max-steps/--num-rollouts-per-step.
    """
    parser = argparse.ArgumentParser(
        description="Train multi-turn PPO/MT-PPO (see CLAUDE.md and docs/phase-7-mt-ppo.md)."
    )
    parser.add_argument("--condition", required=True, choices=["ppo", "mt_ppo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-rollouts-per-step", type=int, default=2)
    return parser.parse_args(argv)


def build_ppo_trainer(
    condition: Condition,
    train_size: int | None,
    config: MTPPOConfig,
) -> MTPPOTrainer:
    """Composition root: real policy+critic, real SearchEnv (hits the live retrieval server),
    real data. Not unit-tested -- this is exactly the integration surface the live smoke test
    validates, same principle as train.py's build_trainer.
    """
    from transformers import AutoTokenizer

    from turn_level_rewards import data

    model = build_policy_and_critic(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = data.load_train_dataset(n=train_size, seed=config.seed)
    return MTPPOTrainer(
        condition=condition,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=config,
    )


def main() -> None:
    args = _parse_args()
    config = build_ppo_config(
        condition=args.condition,
        seed=args.seed,
        max_steps=args.max_steps,
        num_rollouts_per_step=args.num_rollouts_per_step,
    )
    config.run_name = (
        f"{args.condition}-{args.max_steps}steps-"
        f"{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    trainer = build_ppo_trainer(args.condition, args.train_size, config)
    trainer.train()


if __name__ == "__main__":
    main()
```

Replace the inline `__import__('datetime')` with a normal top-of-file `from datetime import datetime`
import and `datetime.now()` call — the inline form above is only written out this way to show
the diff compactly; use the clean top-level import in the real file, matching `train.py`'s own
`from datetime import datetime` convention exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: PASS (21 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add build_ppo_trainer and main() CLI entrypoint"
```

---

### Task 12: `scripts/verify_phase7.py`

**Files:**
- Create: `scripts/verify_phase7.py`

**Interfaces:**
- Consumes: everything built in Tasks 1-11.

- [ ] **Step 1: Write the script**

Create `scripts/verify_phase7.py`, mirroring `scripts/verify_phase4.py`'s exact structure:

```python
#!/usr/bin/env python3
"""Phase 7 exit-criteria check (code portion only).

Mirrors scripts/verify_phase4.py's pattern: prints exactly which check failed, or PASS and exits
0, only if every check below passes. Run this after any change to train_ppo.py.

This only covers the static/testable subset of Phase 7's exit criteria. The live smoke test (both
conditions, manually reading transcripts, checking critic values) is NOT scripted here -- judging
real completion transcripts needs human/agent judgment, not a mechanical check. See
docs/phase-7-mt-ppo.md's exit criteria for that part.

Usage: uv run python scripts/verify_phase7.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PPO_PY = REPO_ROOT / "src" / "turn_level_rewards" / "train_ppo.py"


def _run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def check() -> list[str]:
    failures = []

    code, output = _run("uv", "run", "pytest", "tests/unit/", "-q")
    if code != 0:
        failures.append(f"pytest tests/unit/ failed:\n{output}")

    code, output = _run("uv", "run", "ruff", "check")
    if code != 0:
        failures.append(f"ruff check failed:\n{output}")

    code, output = _run("uv", "run", "ty", "check")
    if code != 0:
        failures.append(f"ty check failed:\n{output}")

    if not TRAIN_PPO_PY.exists():
        failures.append(f"{TRAIN_PPO_PY} does not exist yet.")
        return failures

    from turn_level_rewards.train_ppo import build_ppo_config

    ppo_config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    mt_ppo_config = build_ppo_config("mt_ppo", seed=42, max_steps=2, num_rollouts_per_step=2)

    fixed_checks = {
        "n_max": 4,
        "clip_eps": 0.2,
        "kl_beta": 0.001,
        "policy_lr": 1e-6,
        "critic_lr": 1e-5,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "num_ppo_epochs": 4,
        "value_loss_coef": 0.5,
        "max_completion_length": 2048,
    }
    for config, label in [(ppo_config, "ppo"), (mt_ppo_config, "mt_ppo")]:
        for field, expected in fixed_checks.items():
            actual = getattr(config, field)
            if actual != expected:
                failures.append(
                    f"build_ppo_config({label!r}).{field} == {actual!r}, expected {expected!r}"
                )

    if ppo_config.output_dir != "outputs/ppo":
        failures.append(f"ppo output_dir == {ppo_config.output_dir!r}")
    if mt_ppo_config.output_dir != "outputs/mt_ppo":
        failures.append(f"mt_ppo output_dir == {mt_ppo_config.output_dir!r}")

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 7 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: unit tests, ruff, ty are clean, and build_ppo_config matches the design spec."
    )
    print(
        "This does NOT cover the live smoke test -- run it manually per "
        "docs/phase-7-mt-ppo.md before sign-off."
    )
    sys.exit(0)
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/verify_phase7.py`
Expected: PASS (prints the PASS message above and exits 0)

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_phase7.py
git commit -m "scripts: add verify_phase7.py"
```

---

### Task 13: Live smoke test (manual — both conditions)

**Files:** none (this task runs the code, doesn't change it — unless it surfaces a real bug, in
which case fix it in `train_ppo.py` and commit that fix separately, same as Phase 4's own smoke
test caught three real bugs).

This is the integration validation Tasks 7-11 were explicitly not unit-tested against. Confirm the
retrieval server (`scripts/retrieval_server.py`) is running first (per `docs/phase-1-retrieval-infra.md`).

- [ ] **Step 1: Run the `ppo` condition smoke test**

Run: `uv run python -m turn_level_rewards.train_ppo --condition ppo --train-size 4 --max-steps 2 --num-rollouts-per-step 2`

Expected: completes without error. Manually inspect the run for:
- Real `search` tool calls issued by the model (not silently skipped).
- Real retrieved passages appearing in the tool messages.
- Critic values that are finite (no NaN) and not frozen at a single repeated value across the 2
  steps.
- `place_turn_rewards`'s `R^I` term is confirmed **always 0** for this condition (add a temporary
  print of `per_token_rewards` inside `_collect_batch` during this manual run if needed to verify
  directly, then remove it — don't leave debug prints committed).
- Whether a turn ever produces more than one `tool_call` in the same assistant message. If it
  does, check whether `trl`'s `parse_response`/chat template needs a `tool_call_id` on the
  corresponding tool message to associate the response with the right call (this repo's tool
  messages currently carry only `role`/`name`/`content`, no `tool_call_id` — flagged as an open
  question by Task 7's review, not yet resolved). Record the answer in the Handoff notes either
  way.
- What happens when `n_max` is exhausted while every turn made a tool call (the episode never
  reaches a final-answer turn). Confirm `_collect_batch`'s reward computation
  (`format_reward`/`outcome_reward` on an answerless `completion`) behaves sensibly (a low/negative
  reward, not a crash) rather than assuming it — this edge case was flagged by Task 7's review as
  inherited from the algorithm, not yet observed in a real run.

- [ ] **Step 2: Run the `mt_ppo` condition smoke test**

Run: `uv run python -m turn_level_rewards.train_ppo --condition mt_ppo --train-size 4 --max-steps 2 --num-rollouts-per-step 2`

Expected: same as Step 1, plus: confirm `R^I` is **nonzero** on at least one episode where a gold
supporting-fact title was actually surfaced (cross-check against
`rollout["retrieval_fraction_after_each_turn"]` for that episode).

- [ ] **Step 3: Verify the diagnostic logging artifacts from both runs**

For each of the two runs above, confirm:
- `outputs/{condition}/train_log.jsonl` exists and has exactly 2 lines (one per step), each valid
  JSON with the fields `step`, `loss`, `policy_loss`, `value_loss`, `kl`, `reward`,
  `retrieval_fraction`, `step_elapsed_seconds`, `total_elapsed_seconds`, `eta_seconds`, and an
  `episodes` list with `question`/`format_and_outcome_reward`/`retrieval_fraction`/
  `num_action_tokens` per episode.
- `outputs/{condition}/sample_completions.log` exists and contains at least one real transcript
  (step 0's, since `_SAMPLE_COMPLETION_INTERVAL=10` alone wouldn't trigger within only 2 steps) —
  confirm the question/message text is real model output, not empty or placeholder text.
- The stdout progress line (`step N/max_steps | loss=... reward=... retrieval_fraction=...
  elapsed=...s eta=...s`) printed once per step during the run.
- Re-run the `ppo` condition a second time with the same `--seed` (default 42) and confirm
  `train_log.jsonl`'s `reward`/`retrieval_fraction` values match the first run's step 0 (validates
  `set_seed` actually makes the run repeatable, not just that the file exists) — note any
  divergence in the Handoff notes rather than assuming determinism holds if it doesn't (real GPU
  kernels can have residual nondeterminism `set_seed` alone doesn't eliminate; record what's
  actually observed).

- [ ] **Step 4: Record results in the Handoff notes**

Update `docs/phase-7-mt-ppo.md`'s Handoff notes section (currently `(not yet started)`) with: any
TRL/transformers API surprises found, the real observed per-step wall-clock time, confirmation of
the four exit criteria in that doc, the tool_call_id and answerless-completion findings from Step
1, the repeatability result from Step 3's same-seed re-run, and anything relevant for Phase 7b
(full runs) or Phase 8 (LLM judge) to pick up. This is the same handoff discipline every prior
phase followed.

- [ ] **Step 5: Update CLAUDE.md's roadmap table**

Change Phase 7's Status cell from "Not started" to "**Done**", following the exact style of the
other completed rows (a bolded "Done", then a semicolon-separated summary of what was verified and
a pointer to the Handoff notes for detail).

- [ ] **Step 6: Commit**

```bash
git add docs/phase-7-mt-ppo.md CLAUDE.md
git commit -m "Phase 7 complete: MTPPOTrainer smoke-tested for both ppo and mt_ppo conditions"
```

---

## Self-Review

**Spec coverage:** Every task in `docs/phase-7-mt-ppo.md`'s checklist is covered: `train_ppo.py`
(Tasks 2-11), `tests/unit/test_train_ppo.py` (Tasks 2-6, 11), `scripts/verify_phase7.py`
(Task 12), live smoke test (Task 13). GAE (Task 2) and Eq. 9 placement (Task 3) are unit-tested
per the design spec's testing strategy; the rollout loop, critic construction, and full trainer
integration are explicitly not unit-tested, matching that same section's stated scope.

**Placeholder scan:** No "TBD"/"TODO" in any task's code. `value_loss_coef=0.5` and the KL-as-
direct-loss-term interpretation are both flagged inline as assumptions (matching the design spec),
not presented as paper-derived.

**Type consistency:** `Condition = Literal["ppo", "mt_ppo"]` used consistently from Task 2 onward.
`place_turn_rewards`'s parameter names match between Task 3's definition and Task 9's call site.
`_forward_logprobs_and_values`'s return order `(action_logprobs, action_values)` matches every
call site (Task 9, Task 10). `MTPPOConfig` field names (`n_max`, `clip_eps`, `kl_beta`,
`policy_lr`, `critic_lr`, `gamma`, `gae_lambda`, `num_ppo_epochs`, `value_loss_coef`,
`num_rollouts_per_step`) are used identically across Tasks 5-13, including in
`scripts/verify_phase7.py`.

**Scope check:** This plan covers Phase 7 exactly as scoped (build + smoke test) — no full
training runs, no evaluation, no matplotlib visuals (those are Phase 7b, already stubbed in
`docs/phase-7b-full-ppo-runs.md`).

**Mid-execution amendment:** after Task 7 landed, a request arrived for detailed diagnostic
logging, deterministic repeatability, and long-run progress visibility. Tasks 9/10/13 were
revised in place (before being dispatched) to add `question`/`completion` to `_collect_batch`'s
episode dict, `set_seed`/`train_log.jsonl`/`sample_completions.log`/stdout progress to `train()`,
and a verification step in the live smoke test confirming both log artifacts are populated with
real content and that same-seed reruns actually reproduce.
