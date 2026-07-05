#!/usr/bin/env python3
"""Phase 5 exit-criteria check (static/code portion only).

Mirrors scripts/verify_phase4.py's pattern: prints exactly which check failed, or PASS and exits
0, only if every check below passes. Run this after the build_config/rewards.py changes and
before spending any GPU time on the full training runs.

This only covers the static/testable subset of Phase 5's exit criteria -- the canary dry-run, the
full 300-step runs, and the post-run evidence gate (checkpoint loadability, trackio alerts/curves)
are NOT scripted here; see
docs/superpowers/specs/2026-07-05-phase-5-full-training-runs-design.md's "Verification plan"
section for that layer.

Usage: uv run python scripts/verify_phase5.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    from turn_level_rewards.train import build_config

    outcome_config = build_config(condition="outcome_only", seed=42, max_steps=2, num_generations=2)
    turn_config = build_config(condition="turn_level", seed=42, max_steps=2, num_generations=2)

    fixed_checks = {
        "num_iterations": 2,
        "eval_strategy": "steps",
        "eval_steps": 20,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 3,
    }
    for config, label in [(outcome_config, "outcome_only"), (turn_config, "turn_level")]:
        for field, expected in fixed_checks.items():
            actual = getattr(config, field)
            if actual != expected:
                failures.append(
                    f"build_config({label!r}).{field} == {actual!r}, expected {expected!r}"
                )

    # Regression check for a real bug a live canary run surfaced: eval_strategy="steps" makes
    # GRPOConfig validate per_device_eval_batch_size * num_processes against num_generations, in
    # addition to the already-checked train-side generation_batch_size divisibility. num_generations
    # values that don't divide the HF default per_device_eval_batch_size=8 (e.g. 21, Phase 5's
    # real full-run value) must not raise here.
    try:
        real_scale_config = build_config(
            condition="outcome_only", seed=42, max_steps=2, num_generations=21
        )
    except ValueError as e:
        failures.append(
            f"build_config(num_generations=21) raised {e!r} -- eval-side divisibility broken"
        )
    else:
        if real_scale_config.per_device_eval_batch_size != 21:
            failures.append(
                "build_config(num_generations=21).per_device_eval_batch_size == "
                f"{real_scale_config.per_device_eval_batch_size!r}, expected 21"
            )

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 5's static checks are not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: unit tests, ruff, ty are clean, and build_config's new num_iterations/eval/save "
        "fields match the design spec."
    )
    print(
        "This does NOT cover the canary dry-run, the full training runs, or the post-run "
        "evidence gate -- run those manually per the design spec before sign-off."
    )
    sys.exit(0)
