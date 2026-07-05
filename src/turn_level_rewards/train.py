"""train.py: CLI entrypoint wiring GRPOTrainer + SearchEnv + reward funcs + GRPOConfig + trackio.

See docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md for the full design
rationale (TRL tool-call-loop semantics, NaN-masking gotchas, measured max_completion_length,
and the generation_batch_size/num_generations divisibility constraint).
"""

from typing import Literal

from trl import GRPOConfig

Condition = Literal["outcome_only", "turn_level"]


def build_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_generations: int,
) -> GRPOConfig:
    """Build the GRPOConfig for a training run.

    per_device_train_batch_size is set equal to num_generations, not passed independently --
    GRPOConfig requires generation_batch_size (which defaults to per_device_train_batch_size *
    num_processes * steps_per_generation) to be evenly divisible by num_generations; setting
    them equal satisfies this trivially on a single GPU.
    """
    return GRPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        max_tool_calling_iterations=4,
        beta=0.0,
        max_completion_length=2048,
        logging_steps=1,
        logging_nan_inf_filter=False,
        log_completions=True,
        report_to="trackio",
        project="turn-level-rewards",
        run_name=condition,
    )
