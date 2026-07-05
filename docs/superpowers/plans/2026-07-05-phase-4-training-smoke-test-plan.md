# Phase 4 Training Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/turn_level_rewards/train.py` (CLI entrypoint wiring `GRPOTrainer` +
`SearchEnv` + `get_reward_funcs` + `GRPOConfig` + trackio alerting) so that Phase 4's live smoke
test can be run afterward.

**Architecture:** Three independently-testable pieces plus a thin composition root, per
`docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md`: `build_config` (pure,
unit-tested), `TrackioAlertCallback` (unit-tested with fake `trackio.alert`), and
`build_trainer`/`main` (the real-model/real-network composition root, not unit-tested — that's
what the live smoke test validates instead).

**Tech Stack:** Python 3.13, `trl==1.7.1`, `transformers==5.13.0`, `trackio==0.29.0`, `pytest`,
`argparse` (stdlib, no new CLI dependency).

## Global Constraints

- Every test in `tests/unit/` must be fast, deterministic, and touch no GPU/network/live server
  (CLAUDE.md's Guiding principles, point 4). `train.py`'s module-level imports (`trl`,
  `transformers`, `turn_level_rewards.env`, `turn_level_rewards.data`) must stay side-effect-free
  at import time — none of them call the network or load a model just by being imported.
- `tests/unit/` is the only test tier in this repo — do not add integration/GPU test tiers
  (CLAUDE.md's Repo layout section).
- `per_device_train_batch_size` must equal `num_generations` — confirmed via real `GRPOConfig`
  construction that `generation_batch_size` (which defaults to `per_device_train_batch_size ×
  num_processes × steps_per_generation`) must be evenly divisible by `num_generations`; this holds
  trivially when they're equal. Do not expose `per_device_train_batch_size` as an independent CLI
  flag.
- `max_tool_calling_iterations=4`, `beta=0.0`, `max_completion_length=2048`,
  `logging_steps=1`, `logging_nan_inf_filter=False`, `log_completions=True`,
  `report_to="trackio"`, `project="turn-level-rewards"` are fixed and identical across both
  conditions — only `output_dir` and `run_name` vary by `condition`.
- Do not patch `GRPOTrainer` internals (e.g. to fix `torch.nanmean`'s reward-NaN masking) — same
  "no non-public trainer internals" boundary CLAUDE.md draws around `MT-GRPO`.
- The live smoke test itself (running `train.py` for real against the live retrieval server and a
  downloaded model, reading transcripts) is explicitly **out of scope for this plan's tasks** —
  it happens afterward, manually, in the foreground. Do not have a subagent attempt it.

---

### Task 1: `build_config`

**Files:**
- Create: `src/turn_level_rewards/train.py`
- Test: `tests/unit/test_train.py`

**Interfaces:**
- Produces: `Condition = Literal["outcome_only", "turn_level"]`; `build_config(condition:
  Condition, seed: int, max_steps: int, num_generations: int) -> GRPOConfig`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_train.py`:

```python
"""Fast, GPU-free tests for train.py's build_config and TrackioAlertCallback.

No real GRPOTrainer or model is constructed here -- that integration surface is what the live
smoke test (not tests/unit/) covers instead, per CLAUDE.md's Guiding principles.
"""

from turn_level_rewards.train import build_config


def _build(condition: str, seed: int = 42, max_steps: int = 2, num_generations: int = 2):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.train'`

- [ ] **Step 3: Write the implementation**

Create `src/turn_level_rewards/train.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train.py tests/unit/test_train.py
git commit -m "Add train.py's build_config"
```

---

### Task 2: `TrackioAlertCallback`

**Files:**
- Modify: `src/turn_level_rewards/train.py`
- Modify: `tests/unit/test_train.py`

**Interfaces:**
- Consumes: nothing from Task 1 beyond the module existing.
- Produces: `TrackioAlertCallback` (a `transformers.TrainerCallback` subclass) with a zero-arg
  `__init__` and `on_log(self, args, state, control, logs=None, **kwargs) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_train.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

from turn_level_rewards.train import TrackioAlertCallback


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
    assert mock_alert.call_args.kwargs["title"] == "NaN loss"
    assert control.should_training_stop is True


@patch("turn_level_rewards.train.trackio.alert")
def test_healthy_log_sequence_fires_no_alerts(mock_alert):
    callback = TrackioAlertCallback()
    for step in range(1, 31):
        _log(callback, step, loss=0.5, reward=0.8, frac_reward_zero_std=0.2)

    assert mock_alert.call_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'TrackioAlertCallback' from
'turn_level_rewards.train'`

- [ ] **Step 3: Write the implementation**

Append to `src/turn_level_rewards/train.py` (add `import math`, `from transformers import
TrainerCallback`, and `import trackio` to the top-of-file imports):

```python
import math

import trackio
from transformers import TrainerCallback

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train.py tests/unit/test_train.py
git commit -m "Add TrackioAlertCallback with dead-reward/zero-std/NaN detection"
```

---

### Task 3: CLI parsing, `build_trainer`, `main`

**Files:**
- Modify: `src/turn_level_rewards/train.py`
- Modify: `tests/unit/test_train.py`

**Interfaces:**
- Consumes: `build_config` (Task 1), `TrackioAlertCallback` (Task 2),
  `turn_level_rewards.data.load_train_dataset`/`load_eval_dataset` (Phase 3, already merged),
  `turn_level_rewards.env.SearchEnv` (Phase 2, already merged),
  `turn_level_rewards.rewards.get_reward_funcs` (Phase 2, already merged).
- Produces: `_parse_args(argv: list[str] | None = None) -> argparse.Namespace`; `build_trainer(
  condition: Condition, train_size: int | None, eval_size: int | None, config: GRPOConfig) ->
  GRPOTrainer`; `main() -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_train.py`:

```python
import pytest

from turn_level_rewards.train import _parse_args


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: FAIL with `ImportError: cannot import name '_parse_args' from
'turn_level_rewards.train'`

- [ ] **Step 3: Write the implementation**

Append to `src/turn_level_rewards/train.py` (add `import argparse`, `from trl import
GRPOConfig, GRPOTrainer` (extend the existing `trl` import), and `from turn_level_rewards import
data`, `from turn_level_rewards.env import SearchEnv`, `from turn_level_rewards.rewards import
get_reward_funcs` to the top-of-file imports):

```python
import argparse

from trl import GRPOTrainer

from turn_level_rewards import data
from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import get_reward_funcs


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
        environment_factory=SearchEnv,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: PASS (13 tests total)

Also run:
```bash
uv run ruff check
uv run ty check
```
Expected: both clean. Fix import order/typing issues if either flags anything (e.g. `ruff`'s
`I001` isort rule may want the appended imports reordered/merged with the top-of-file block from
Tasks 1-2 — consolidate into one clean import block at the top of the file rather than leaving
imports scattered across three append points).

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train.py tests/unit/test_train.py
git commit -m "Add train.py's CLI, build_trainer, and main entrypoint"
```

---

### Task 4: Phase 4 exit-criteria script (code portion)

**Files:**
- Create: `scripts/verify_phase4.py`

**Interfaces:**
- Consumes: `build_config`, `TrackioAlertCallback` (Tasks 1-2).
- Produces: nothing consumed by later tasks — this is the last task in this plan.

- [ ] **Step 1: Write the script**

Create `scripts/verify_phase4.py`:

```python
#!/usr/bin/env python3
"""Phase 4 exit-criteria check (code portion only).

Mirrors scripts/verify_phase2.py/verify_phase3.py's pattern: prints exactly which check failed,
or PASS and exits 0, only if every check below passes. Run this after any change to train.py.

This only covers the static/testable subset of Phase 4's exit criteria. The live smoke test
(both conditions, manually reading transcripts, checking trackio alerts) is NOT scripted here --
judging real completion transcripts and trackio dashboards needs human/agent judgment, not a
mechanical check. See docs/phase-4-training-smoke-test.md's exit criteria for that part.

Usage: uv run python scripts/verify_phase4.py
"""

import inspect
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "src" / "turn_level_rewards" / "train.py"


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

    if not TRAIN_PY.exists():
        failures.append(f"{TRAIN_PY} does not exist yet.")
        return failures

    from transformers import TrainerCallback

    from turn_level_rewards.train import TrackioAlertCallback, build_config

    if "per_device_train_batch_size" in inspect.signature(build_config).parameters:
        failures.append(
            "build_config accepts per_device_train_batch_size as a separate parameter -- it "
            "must be derived internally as equal to num_generations, not exposed independently "
            "(see the design spec's generation_batch_size divisibility finding)."
        )

    outcome_config = build_config(condition="outcome_only", seed=42, max_steps=2, num_generations=2)
    turn_config = build_config(condition="turn_level", seed=42, max_steps=2, num_generations=2)

    fixed_checks = {
        "max_tool_calling_iterations": 4,
        "beta": 0.0,
        "max_completion_length": 2048,
        "logging_steps": 1,
        "logging_nan_inf_filter": False,
        "log_completions": True,
        "project": "turn-level-rewards",
    }
    for config, label in [(outcome_config, "outcome_only"), (turn_config, "turn_level")]:
        for field, expected in fixed_checks.items():
            actual = getattr(config, field)
            if actual != expected:
                failures.append(
                    f"build_config({label!r}).{field} == {actual!r}, expected {expected!r}"
                )
        if config.report_to != ["trackio"]:
            failures.append(
                f"build_config({label!r}).report_to == {config.report_to!r}, expected ['trackio']"
            )
        if config.per_device_train_batch_size != config.num_generations:
            failures.append(
                f"build_config({label!r}).per_device_train_batch_size "
                f"({config.per_device_train_batch_size}) != num_generations "
                f"({config.num_generations})"
            )
        if config.generation_batch_size % config.num_generations != 0:
            failures.append(
                f"build_config({label!r}).generation_batch_size not divisible by num_generations"
            )

    if outcome_config.output_dir != "outputs/outcome_only":
        failures.append(f"outcome_only output_dir == {outcome_config.output_dir!r}")
    if turn_config.output_dir != "outputs/turn_level":
        failures.append(f"turn_level output_dir == {turn_config.output_dir!r}")

    if not issubclass(TrackioAlertCallback, TrainerCallback):
        failures.append("TrackioAlertCallback does not subclass transformers.TrainerCallback")

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 4 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: unit tests, ruff, ty are clean, and build_config/TrackioAlertCallback match the "
        "design spec."
    )
    print(
        "This does NOT cover the live smoke test -- run it manually per "
        "docs/phase-4-training-smoke-test.md and the Phase 4 design spec before sign-off."
    )
    sys.exit(0)
```

- [ ] **Step 2: Run it to verify PASS**

Run: `uv run python scripts/verify_phase4.py`
Expected: exits 0, prints the `PASS` message above.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_phase4.py
git commit -m "Add Phase 4 exit-criteria validation script (code portion)"
```

---

## After this plan: not part of any task above

Once all 4 tasks are reviewed and merged, the live smoke test happens manually in the foreground
(not delegated to a subagent) — run both conditions, read the `rich`-printed transcripts, check
`trackio list alerts --project turn-level-rewards --json`, per the design spec's "Smoke test
execution" section. Only after that succeeds should `docs/phase-4-training-smoke-test.md`'s
checkboxes and Handoff notes, and CLAUDE.md's Roadmap table, be updated to mark Phase 4 done —
this plan's tasks alone don't satisfy Phase 4's real exit criteria, which include the smoke test.
