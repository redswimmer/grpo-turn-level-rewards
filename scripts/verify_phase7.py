#!/usr/bin/env python3
"""Phase 7 exit-criteria check (code portion only).

Mirrors scripts/verify_phase4.py's pattern: prints exactly which check failed, or PASS and exits
0, only if every check below passes. Run this after any change to train_ppo.py.

This only covers the static/testable subset of Phase 7's exit criteria. The live smoke test (both
conditions, manually reading transcripts, checking critic values) is NOT scripted here -- judging
real completion transcripts needs human/agent judgment, not a mechanical check. See
docs/phase-7-mt-ppo.md's exit criteria for that part.

Usage: uv run python scripts/verify_phase7.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PPO_PY = REPO_ROOT / "src" / "turn_level_rewards" / "train_ppo.py"


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

    if not TRAIN_PPO_PY.exists():
        failures.append(f"{TRAIN_PPO_PY} does not exist yet.")
        return failures

    from turn_level_rewards.train_ppo import build_ppo_config

    ppo_config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    mt_ppo_config = build_ppo_config("mt_ppo", seed=42, max_steps=2, num_rollouts_per_step=2)

    fixed_checks = {
        "n_max": 4,
        "clip_eps": 0.2,
        "kl_beta": 0.001,
        "policy_lr": 1e-6,
        "critic_lr": 1e-5,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "num_ppo_epochs": 4,
        "value_loss_coef": 0.5,
        "max_completion_length": 2048,
    }
    for config, label in [(ppo_config, "ppo"), (mt_ppo_config, "mt_ppo")]:
        for field, expected in fixed_checks.items():
            actual = getattr(config, field)
            if actual != expected:
                failures.append(
                    f"build_ppo_config({label!r}).{field} == {actual!r}, expected {expected!r}"
                )

    if ppo_config.output_dir != "outputs/ppo":
        failures.append(f"ppo output_dir == {ppo_config.output_dir!r}")
    if mt_ppo_config.output_dir != "outputs/mt_ppo":
        failures.append(f"mt_ppo output_dir == {mt_ppo_config.output_dir!r}")

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 7 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: unit tests, ruff, ty are clean, and build_ppo_config matches the design spec.")
    print(
        "This does NOT cover the live smoke test -- run it manually per "
        "docs/phase-7-mt-ppo.md before sign-off."
    )
    sys.exit(0)
