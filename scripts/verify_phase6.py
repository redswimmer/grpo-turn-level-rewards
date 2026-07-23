#!/usr/bin/env python3
"""Phase 6 exit-criteria check (static/code portion only).

Mirrors scripts/verify_phase5.py's pattern. This only covers the static/testable subset --
the canary dry-run, the real full evaluations, the comparison plots, and the more-training
decision are NOT scripted here; see docs/phase-6-evaluation-comparison.md's "Exit criteria"
section for that layer.

Usage: uv run python scripts/verify_phase6.py
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

    from turn_level_rewards.evaluate import build_eval_config

    fixed_checks = {
        "num_generations": 2,
        "max_tool_calling_iterations": 4,
        "beta": 0.0,
        "max_completion_length": 2048,
        "report_to": [],
    }
    for condition in ("outcome_only", "turn_level"):
        config = build_eval_config(condition, eval_batch_size=8)
        for field, expected in fixed_checks.items():
            actual = getattr(config, field)
            if actual != expected:
                failures.append(
                    f"build_eval_config({condition!r}).{field} == {actual!r}, expected {expected!r}"
                )

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 6's static checks are not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: unit tests, ruff, ty are clean, and build_eval_config's fixed fields match "
        "the design spec."
    )
    print(
        "This does NOT cover the canary dry-run, the real full evaluations, the comparison "
        "plots, or the more-training decision -- run those manually per "
        "docs/phase-6-evaluation-comparison.md before sign-off."
    )
    sys.exit(0)
