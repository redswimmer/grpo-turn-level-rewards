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
        "gradient_checkpointing": True,
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
        assert config.generation_batch_size is not None
        assert config.num_generations is not None
        if config.generation_batch_size % config.num_generations != 0:
            failures.append(
                f"build_config({label!r}).generation_batch_size not divisible by num_generations"
            )

    if outcome_config.output_dir != "outputs/outcome_only":
        failures.append(f"outcome_only output_dir == {outcome_config.output_dir!r}")
    if turn_config.output_dir != "outputs/turn_level":
        failures.append(f"turn_level output_dir == {turn_config.output_dir!r}")

    if outcome_config.run_name != "outcome_only":
        failures.append(f"outcome_only run_name == {outcome_config.run_name!r}")
    if turn_config.run_name != "turn_level":
        failures.append(f"turn_level run_name == {turn_config.run_name!r}")

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
