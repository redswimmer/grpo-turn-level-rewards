#!/usr/bin/env python3
"""Phase 3 exit-criteria check.

Mirrors scripts/verify_phase2.py's pattern: prints exactly which check failed, or PASS and exits
0, only if every check below passes. Run this after any change to data.py.

Usage: uv run python scripts/verify_phase3.py
"""

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PY = REPO_ROOT / "src" / "turn_level_rewards" / "data.py"


def _run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def _has_injectable_loader_param(tree: ast.AST, func_name: str) -> bool:
    """True if `func_name`'s def has a keyword-only param named `load_dataset_fn`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "load_dataset_fn" in [arg.arg for arg in node.args.kwonlyargs]
    return False


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

    if not DATA_PY.exists():
        failures.append(f"{DATA_PY} does not exist yet.")
    else:
        tree = ast.parse(DATA_PY.read_text())
        for func_name in ("load_train_dataset", "load_eval_dataset"):
            if not _has_injectable_loader_param(tree, func_name):
                failures.append(
                    f"{func_name} in {DATA_PY} has no keyword-only `load_dataset_fn` param -- "
                    "the dataset-loading seam must be injectable, not hardcoded."
                )

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 3 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: unit tests, ruff, and ty are clean, and both loaders have an injectable seam.")
    print("Run the manual real-data check (see docs/phase-3-data-pipeline.md) before sign-off.")
    sys.exit(0)
