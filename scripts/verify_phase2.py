#!/usr/bin/env python3
"""Phase 2 exit-criteria check.

Mirrors scripts/verify_retrieval.py's pattern: prints exactly which check failed, or PASS and
exits 0, only if every check below passes. Run this after any change to metrics.py/env.py/
rewards.py, and again before marking Phase 2 done in docs/phase-2-core-library.md.

Usage: uv run python scripts/verify_phase2.py
"""

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PY = REPO_ROOT / "src" / "turn_level_rewards" / "env.py"


def _run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def _is_requests_or_httpx_call(node: ast.AST) -> bool:
    """True if `node` is a Call whose callee is an attribute access rooted at requests/httpx.

    Matches `requests.post(...)`, `httpx.post(...)`, `httpx.Client().post(...)`, etc. -- anything
    whose attribute-access chain bottoms out at a bare `requests`/`httpx` name.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    root = node.func.value
    while isinstance(root, ast.Attribute):
        root = root.value
    return isinstance(root, ast.Name) and root.id in {"requests", "httpx"}


def _find_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef | None:
    """Find `def method_name` directly inside `class class_name` (not nested classes)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


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
        tree = ast.parse(env_source, filename=str(ENV_PY))

        any_call_in_file = any(_is_requests_or_httpx_call(n) for n in ast.walk(tree))
        if not any_call_in_file:
            failures.append(
                f"No requests.post/httpx call found anywhere in {ENV_PY} -- retrieval isn't wired."
            )

        search_method = _find_method(tree, "SearchEnv", "search")
        if search_method is None:
            failures.append(
                f"Could not find `def search` inside `class SearchEnv` in {ENV_PY} -- cannot "
                "verify the retrieval seam stays injectable."
            )
        else:
            inline_calls = [n for n in ast.walk(search_method) if _is_requests_or_httpx_call(n)]
            if inline_calls:
                failures.append(
                    f"Found a requests.post/httpx call inlined directly inside "
                    f"SearchEnv.search() in {ENV_PY} -- SearchEnv.search() must only call "
                    "self._retrieve_fn, never requests/httpx directly. Move the HTTP call back "
                    "into a factory function (e.g. _default_retrieve_fn) and inject it via "
                    "self._retrieve_fn so tests never hit the network."
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
