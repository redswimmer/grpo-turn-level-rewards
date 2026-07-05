"""train.py: CLI entrypoint wiring GRPOTrainer + SearchEnv + reward funcs + GRPOConfig + trackio.

See docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md for the full design
rationale (TRL tool-call-loop semantics, NaN-masking gotchas, measured max_completion_length,
and the generation_batch_size/num_generations divisibility constraint).
"""

import math
from typing import Literal

import trackio
from transformers import TrainerCallback
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


_DEAD_REWARD_STEP_THRESHOLD = 20
_ZERO_STD_STREAK_THRESHOLD = 20


class TrackioAlertCallback(TrainerCallback):
    """Fires trackio alerts for silent-failure modes a clean exit code wouldn't catch.

    See docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md's "Alert
    callback" section for the reasoning behind each threshold and the re-arming behavior.
    Reward-side NaN detection is intentionally not implemented here: GRPOTrainer aggregates the
    logged `reward` metric via torch.nanmean, which silently drops NaN entries rather than
    propagating them, and this repo's own reward functions have no real numerical path to NaN.
    """

    def __init__(self) -> None:
        self._reward_ever_nonzero = False
        self._dead_reward_alerted = False
        self._zero_std_streak = 0
        self._zero_std_alerted = False

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return

        loss = logs.get("loss")
        if loss is not None and math.isnan(loss):
            trackio.alert(
                title="NaN loss",
                text=f"Loss is NaN at step {state.global_step} -- stopping training.",
                level=trackio.AlertLevel.ERROR,
            )
            control.should_training_stop = True
            return

        reward = logs.get("reward")
        if reward is not None:
            if reward != 0.0:
                self._reward_ever_nonzero = True
            if (
                not self._reward_ever_nonzero
                and not self._dead_reward_alerted
                and state.global_step > _DEAD_REWARD_STEP_THRESHOLD
            ):
                trackio.alert(
                    title="Dead reward",
                    text=(
                        f"Reward has been exactly 0.0 for all {state.global_step} steps so far "
                        "-- possible miswired reward function or tool-calling loop."
                    ),
                    level=trackio.AlertLevel.ERROR,
                )
                self._dead_reward_alerted = True

        frac_zero_std = logs.get("frac_reward_zero_std")
        if frac_zero_std is not None:
            if frac_zero_std == 1.0:
                self._zero_std_streak += 1
            else:
                self._zero_std_streak = 0
                self._zero_std_alerted = False
            if self._zero_std_streak >= _ZERO_STD_STREAK_THRESHOLD and not self._zero_std_alerted:
                trackio.alert(
                    title="No learning signal",
                    text=(
                        f"frac_reward_zero_std has been 1.0 for {self._zero_std_streak} "
                        "consecutive logged steps -- every group is scoring identically, so "
                        "the policy gradient is zero."
                    ),
                    level=trackio.AlertLevel.WARN,
                )
                self._zero_std_alerted = True
