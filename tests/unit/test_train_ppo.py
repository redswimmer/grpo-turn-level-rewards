"""Fast, GPU-free tests for train_ppo.py's pure functions and config builder.

No real MTPPOTrainer, model, or GPU is constructed here -- the rollout loop and critic
construction require a real model/chat-template, which is exactly what the live smoke test
(not tests/unit/) validates instead, per CLAUDE.md's Guiding principles.
"""

from turn_level_rewards.train_ppo import compute_gae, place_turn_rewards


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
