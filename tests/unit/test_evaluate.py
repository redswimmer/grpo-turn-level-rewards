"""Fast, GPU-free tests for evaluate.py's build_eval_config and CLI parsing.

No real GRPOTrainer, model, or checkpoint is constructed here -- that integration surface is
what the real canary/full evaluation run (not tests/unit/) covers instead, per CLAUDE.md's
Guiding principles.
"""

import pytest
from trl import GRPOConfig
from turn_level_rewards.evaluate import Condition, _parse_args, build_eval_config


def _build(condition: Condition, eval_batch_size: int = 8) -> GRPOConfig:
    return build_eval_config(condition=condition, eval_batch_size=eval_batch_size)


def test_build_eval_config_fixed_fields_identical_across_conditions():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    for config in (outcome_config, turn_config):
        assert config.num_generations == 2
        assert config.max_tool_calling_iterations == 4
        assert config.beta == 0.0
        assert config.max_completion_length == 2048
        assert config.report_to == []


def test_build_eval_config_batch_size_wiring():
    config = _build("outcome_only", eval_batch_size=8)

    assert config.per_device_train_batch_size == 8
    assert config.per_device_eval_batch_size == 8
    assert config.generation_batch_size == 8
    assert config.num_generations is not None
    assert config.generation_batch_size % config.num_generations == 0


def test_build_eval_config_output_dir_differs_by_condition():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    assert outcome_config.output_dir == "outputs/outcome_only/eval-scratch"
    assert turn_config.output_dir == "outputs/turn_level/eval-scratch"


def test_build_eval_config_rejects_odd_eval_batch_size():
    with pytest.raises(ValueError, match="even"):
        build_eval_config(condition="outcome_only", eval_batch_size=3)


def test_parse_args_defaults():
    args = _parse_args(
        ["--condition", "outcome_only", "--checkpoint", "outputs/outcome_only/checkpoint-300"]
    )

    assert args.condition == "outcome_only"
    assert args.checkpoint == "outputs/outcome_only/checkpoint-300"
    assert args.eval_batch_size == 2
    assert args.eval_size == 4
    assert args.output is None


def test_parse_args_condition_required():
    with pytest.raises(SystemExit):
        _parse_args(["--checkpoint", "outputs/outcome_only/checkpoint-300"])


def test_parse_args_checkpoint_required():
    with pytest.raises(SystemExit):
        _parse_args(["--condition", "outcome_only"])


def test_parse_args_condition_choices_enforced():
    with pytest.raises(SystemExit):
        _parse_args(["--condition", "bogus", "--checkpoint", "x"])


def test_parse_args_overrides():
    args = _parse_args(
        [
            "--condition",
            "turn_level",
            "--checkpoint",
            "outputs/turn_level/checkpoint-300",
            "--eval-batch-size",
            "32",
            "--eval-size",
            "7405",
            "--output",
            "results/turn_level_eval_metrics.json",
        ]
    )

    assert args.condition == "turn_level"
    assert args.checkpoint == "outputs/turn_level/checkpoint-300"
    assert args.eval_batch_size == 32
    assert args.eval_size == 7405
    assert args.output == "results/turn_level_eval_metrics.json"
