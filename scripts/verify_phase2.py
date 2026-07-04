#!/usr/bin/env python3
"""Phase 2 exit-criteria check.

Mirrors scripts/verify_retrieval.py's pattern: prints exactly which check failed, or PASS and
exits 0, only if every check below passes. Run this after any change to metrics.py/env.py/
rewards.py, and again before marking Phase 2 done in docs/phase-2-core-library.md.

Usage: uv run python scripts/verify_phase2.py
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PY = REPO_ROOT / "src" / "turn_level_rewards" / "env.py"


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

    if not ENV_PY.exists():
        failures.append(f"{ENV_PY} does not exist yet.")
    else:
        env_source = ENV_PY.read_text()
        occurrences = len(re.findall(r"requests\.post\(|httpx\.", env_source))
        if occurrences == 0:
            failures.append(
                f"No requests.post/httpx call found in {ENV_PY} -- retrieval isn't wired."
            )
        elif occurrences > 1:
            failures.append(
                f"Found {occurrences} requests.post/httpx call sites in {ENV_PY} -- expected "
                "exactly one, inside the default retrieve_fn factory. SearchEnv.search() must "
                "only call self._retrieve_fn, never requests/httpx directly."
            )

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 2 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: unit tests, ruff, and ty are clean, and the retrieval seam is genuinely injectable."
    )
    print("Phase 2 exit criteria met -- safe to start Phase 3.")
    sys.exit(0)
