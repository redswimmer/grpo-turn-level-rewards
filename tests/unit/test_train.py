"""Fast, GPU-free tests for train.py's build_config and TrackioAlertCallback.

No real GRPOTrainer or model is constructed here -- that integration surface is what the live
smoke test (not tests/unit/) covers instead, per CLAUDE.md's Guiding principles.
"""

from turn_level_rewards.train import Condition, build_config


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


def test_build_config_per_device_train_batch_size_matches_num_generations():
    config = _build("outcome_only", num_generations=8)
    assert config.per_device_train_batch_size == 8
    assert config.generation_batch_size % config.num_generations == 0
