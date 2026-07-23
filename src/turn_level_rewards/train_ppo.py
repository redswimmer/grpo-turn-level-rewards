"""train_ppo.py: custom multi-turn PPO trainer (MTPPOTrainer) for ppo/mt_ppo conditions.

Built directly on transformers.Trainer, not GRPOTrainer/PPOTrainer -- TRL's PPOTrainer has no
multi-turn tool-calling support (confirmed fresh against the installed 1.7.1 and upstream's
dev branch, re-verified 2026-07-23; see
docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md). Reuses SearchEnv/rewards.py/data.py
unmodified. See CLAUDE.md's Goal section and docs/phase-7-mt-ppo.md for the full design.
"""

from typing import Literal

import torch

from turn_level_rewards.rewards import TURN_REWARD_SCALE

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
