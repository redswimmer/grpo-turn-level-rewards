"""Fast, GPU-free tests for train.py's build_config and TrackioAlertCallback.

No real GRPOTrainer or model is constructed here -- that integration surface is what the live
smoke test (not tests/unit/) covers instead, per CLAUDE.md's Guiding principles.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from turn_level_rewards.train import (
    Condition,
    TrackioAlertCallback,
    _parse_args,
    _train_micro_batch_size,
    build_config,
)


def _build(condition: Condition, seed: int = 42, max_steps: int = 2, num_generations: int = 2):
    return build_config(
        condition=condition, seed=seed, max_steps=max_steps, num_generations=num_generations
    )


def test_build_config_fixed_hyperparameters_identical_across_conditions():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    assert outcome_config.max_tool_calling_iterations == 4
    assert turn_config.max_tool_calling_iterations == 4
    assert outcome_config.beta == 0.0
    assert turn_config.beta == 0.0
    assert outcome_config.max_completion_length == 2048
    assert turn_config.max_completion_length == 2048
    assert outcome_config.logging_steps == 1
    assert turn_config.logging_steps == 1
    assert outcome_config.logging_nan_inf_filter is False
    assert turn_config.logging_nan_inf_filter is False
    assert outcome_config.log_completions is True
    assert turn_config.log_completions is True
    assert outcome_config.gradient_checkpointing is True
    assert turn_config.gradient_checkpointing is True
    assert outcome_config.report_to == ["trackio"]
    assert turn_config.report_to == ["trackio"]
    assert outcome_config.project == "turn-level-rewards"
    assert turn_config.project == "turn-level-rewards"


def test_build_config_output_dir_and_run_name_differ_by_condition():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    assert outcome_config.output_dir == "outputs/outcome_only"
    assert turn_config.output_dir == "outputs/turn_level"
    assert outcome_config.run_name == "outcome_only"
    assert turn_config.run_name == "turn_level"


def test_build_config_only_condition_derived_fields_differ():
    outcome_config = _build("outcome_only", seed=7, max_steps=3, num_generations=4)
    turn_config = _build("turn_level", seed=7, max_steps=3, num_generations=4)

    outcome_dict = outcome_config.to_dict()
    turn_dict = turn_config.to_dict()
    differing_fields = {key for key in outcome_dict if outcome_dict[key] != turn_dict.get(key)}

    assert differing_fields == {"output_dir", "run_name"}


def test_train_micro_batch_size_returns_largest_divisor_within_an_explicit_cap():
    assert _train_micro_batch_size(21, cap=3) == 3  # divides evenly
    assert _train_micro_batch_size(8, cap=3) == 2  # 3 doesn't divide 8; falls back a divisor
    assert _train_micro_batch_size(7, cap=3) == 1  # prime > cap; worst-case single-sequence chunks


def test_train_micro_batch_size_defaults_to_the_module_cap_of_one():
    """The real, currently-effective cap: 3 still OOMed on a real canary run's backward pass (a
    much smaller shortfall than the original unchunked-at-21 crash, confirmed not a fragmentation
    artifact), so 1 -- fully sequential, single-sequence chunks -- is what build_config actually
    uses today.
    """
    assert _train_micro_batch_size(2) == 1
    assert _train_micro_batch_size(21) == 1


def test_build_config_train_batch_size_is_capped_micro_batch_with_matching_grad_accumulation():
    """Regression test #1: a real canary run at num_generations=21 with per_device_train_batch_size
    equal to num_generations (no chunking) tried to allocate 28.29 GiB in a single logits-to-fp32
    conversion and OOMed a 24GB GPU. per_device_train_batch_size must stay capped.

    Regression test #2: a real 6300-step run collapsed into a fixed, zero-variance reward within
    ~300 steps because the difference was made up via steps_per_generation directly (leaving
    gradient_accumulation_steps at its default of 1) -- each memory-safe micro-batch triggered its
    own independent optimizer step instead of being combined into one properly-averaged update per
    rollout group. gradient_accumulation_steps must make up the difference instead (steps_per_generation
    then defaults to match it, per TRL's own documented behavior), so the full rollout group size
    (generation_batch_size) still equals num_generations exactly, via real accumulation.
    """
    config = _build("outcome_only", num_generations=21)

    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 21
    assert config.steps_per_generation == 21
    assert config.generation_batch_size == 21
    assert config.generation_batch_size % config.num_generations == 0


def test_build_config_periodic_eval_stays_disabled():
    """Periodic in-training eval is deliberately NOT enabled (see build_config's docstring for the
    environments-pool incompatibility this ran into, and the upstream TRL PR that fixes it). This
    guards against silently re-enabling it without also revisiting the trl pin.
    """
    config = _build("outcome_only")

    assert config.eval_strategy == "no"


def test_build_config_num_iterations_and_save_fields():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    for config in (outcome_config, turn_config):
        assert config.num_iterations == 2
        assert config.save_strategy == "steps"
        assert config.save_steps == 50
        assert config.save_total_limit == 3


def _log(callback, step, **fields):
    state = SimpleNamespace(global_step=step)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(args=None, state=state, control=control, logs=fields)
    return control


@patch("turn_level_rewards.train.trackio.alert")
def test_dead_reward_alert_fires_once_past_step_20(mock_alert):
    callback = TrackioAlertCallback()
    for step in range(1, 26):
        _log(callback, step, reward=0.0, frac_reward_zero_std=0.0)

    assert mock_alert.call_count == 1
    assert mock_alert.call_args.kwargs["title"] == "Dead reward"


@patch("turn_level_rewards.train.trackio.alert")
def test_dead_reward_alert_does_not_fire_if_reward_was_ever_nonzero(mock_alert):
    callback = TrackioAlertCallback()
    _log(callback, 1, reward=0.5, frac_reward_zero_std=0.0)
    for step in range(2, 26):
        _log(callback, step, reward=0.0, frac_reward_zero_std=0.0)

    assert mock_alert.call_count == 0


@patch("turn_level_rewards.train.trackio.alert")
def test_zero_std_streak_fires_once_and_rearms(mock_alert):
    callback = TrackioAlertCallback()
    for step in range(1, 21):
        _log(callback, step, reward=0.5, frac_reward_zero_std=1.0)

    assert mock_alert.call_count == 1
    assert mock_alert.call_args.kwargs["title"] == "No learning signal"

    _log(callback, 21, reward=0.5, frac_reward_zero_std=0.5)  # streak breaks
    for step in range(22, 42):
        _log(callback, step, reward=0.5, frac_reward_zero_std=1.0)  # streak resumes, re-trips

    assert mock_alert.call_count == 2


@patch("turn_level_rewards.train.trackio.alert")
def test_nan_loss_fires_immediately_and_stops_training(mock_alert):
    callback = TrackioAlertCallback()
    control = _log(callback, 5, loss=float("nan"), reward=0.5, frac_reward_zero_std=0.5)

    assert mock_alert.call_count == 1
    assert mock_alert.call_args.kwargs["title"] == "Non-finite loss"
    assert control.should_training_stop is True


@patch("turn_level_rewards.train.trackio.alert")
def test_inf_loss_fires_immediately_and_stops_training(mock_alert):
    callback = TrackioAlertCallback()
    control = _log(callback, 5, loss=float("inf"), reward=0.5, frac_reward_zero_std=0.5)

    assert mock_alert.call_count == 1
    assert mock_alert.call_args.kwargs["title"] == "Non-finite loss"
    assert control.should_training_stop is True

    control = _log(
        TrackioAlertCallback(), 5, loss=float("-inf"), reward=0.5, frac_reward_zero_std=0.5
    )
    assert control.should_training_stop is True


@patch("turn_level_rewards.train.trackio.alert")
def test_healthy_log_sequence_fires_no_alerts(mock_alert):
    callback = TrackioAlertCallback()
    for step in range(1, 31):
        _log(callback, step, loss=0.5, reward=0.8, frac_reward_zero_std=0.2)

    assert mock_alert.call_count == 0


def test_parse_args_defaults():
    args = _parse_args(["--condition", "outcome_only"])

    assert args.condition == "outcome_only"
    assert args.seed == 42
    assert args.train_size == 8
    assert args.eval_size == 8
    assert args.max_steps == 2
    assert args.num_generations == 2


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
            "turn_level",
            "--seed",
            "7",
            "--train-size",
            "90447",
            "--eval-size",
            "200",
            "--max-steps",
            "500",
            "--num-generations",
            "8",
        ]
    )

    assert args.condition == "turn_level"
    assert args.seed == 7
    assert args.train_size == 90447
    assert args.eval_size == 200
    assert args.max_steps == 500
    assert args.num_generations == 8
