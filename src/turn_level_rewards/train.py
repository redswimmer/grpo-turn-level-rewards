"""train.py: CLI entrypoint wiring GRPOTrainer + SearchEnv + reward funcs + GRPOConfig + trackio.

See docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md for the full design
rationale (TRL tool-call-loop semantics, NaN-masking gotchas, measured max_completion_length,
and the generation_batch_size/num_generations divisibility constraint).
"""

import argparse
import math
from typing import Literal

import trackio
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from turn_level_rewards import data
from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import get_reward_funcs

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
        if loss is not None and not math.isfinite(loss):
            trackio.alert(
                title="Non-finite loss",
                text=f"Loss is {loss} at step {state.global_step} -- stopping training.",
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse train.py's CLI arguments.

    The bare invocation (just --condition) IS Phase 4's smoke test -- see the design spec's
    "CLI" section. Full-scale runs must explicitly override every size/step/generation flag.
    """
    parser = argparse.ArgumentParser(
        description="Train GRPO with outcome-only or turn-level reward (see CLAUDE.md)."
    )
    parser.add_argument("--condition", required=True, choices=["outcome_only", "turn_level"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    return parser.parse_args(argv)


def build_trainer(
    condition: Condition,
    train_size: int | None,
    eval_size: int | None,
    config: GRPOConfig,
) -> GRPOTrainer:
    """Composition root: real model, real SearchEnv (hits the live retrieval server), real data.

    Not unit-tested -- this is exactly the integration surface the live smoke test validates.
    """
    return GRPOTrainer(
        model="Qwen/Qwen3.5-0.8B",
        reward_funcs=get_reward_funcs(condition),
        args=config,
        train_dataset=data.load_train_dataset(n=train_size, seed=config.seed),
        eval_dataset=data.load_eval_dataset(n=eval_size, seed=config.seed),
        # SearchEnv.reset requires `metadata`, which is stricter than TRL's `_SupportsReset`
        # protocol (bare **kwargs). This is safe in practice: per CLAUDE.md's "TRL mechanics"
        # section, TRL always calls reset() with the entire sampled dataset row as kwargs, and
        # every row (both loaders in data.py) always includes a `metadata` column.
        environment_factory=SearchEnv,  # ty: ignore[invalid-argument-type]
        callbacks=[TrackioAlertCallback()],
    )


def main() -> None:
    args = _parse_args()
    config = build_config(
        condition=args.condition,
        seed=args.seed,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
    )
    trainer = build_trainer(args.condition, args.train_size, args.eval_size, config)
    trainer.train()


if __name__ == "__main__":
    main()
