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
