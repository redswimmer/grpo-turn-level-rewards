# Phase 3 Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/turn_level_rewards/data.py` so `load_train_dataset`/`load_eval_dataset` produce rows in one shared column contract (`prompt`, `question`, `golden_answers`, `metadata`) that `env.py`/`rewards.py` (Phase 2) can consume unmodified regardless of which source dataset a row came from, with a fast `tests/unit/test_data.py` suite and a `verify_phase3.py` gate.

**Architecture:** A single module, `data.py`, with two public loaders (`load_train_dataset`, `load_eval_dataset`) that each: load via an injectable `load_dataset_fn` seam, filter/shuffle/select, then `.map()` a private per-source row-formatting function that both funnel through one shared prompt-building helper. No other Phase 2 module is touched.

**Tech Stack:** Python 3.13, `datasets` (already a dependency), `pytest`.

## Global Constraints

- Output column contract, identical for both loaders: `prompt` (`list[{"role", "content"}]`),
  `question` (`str`), `golden_answers` (`list[str]`), `metadata` (`dict` with `type`, `level`,
  `supporting_facts: {"title", "sent_id"}`, `context: {"title", "sentences"}`).
- `load_train_dataset`/`load_eval_dataset` each take a keyword-only `load_dataset_fn` parameter
  defaulting to the real `datasets.load_dataset`, per CLAUDE.md's dependency-inversion principle
  (network I/O must be an injected seam) — tests never hit the network.
- Never load the `test` split of `PeterJinGo/nq_hotpotqa_train` (confirmed broken schema) — hardcode
  `split="train"` for the train loader and `split="validation"` for the eval loader.
- Train source: `PeterJinGo/nq_hotpotqa_train`, `"default"` config, `split="train"`, filtered to
  `data_source == "hotpotqa"`.
- Eval source: `hotpotqa/hotpot_qa`, `"distractor"` config, `split="validation"`.
- System prompt must literally state the soft search-count limit "at most 2 searches" (Phase 4's
  `max_tool_calling_iterations` hard cutoff is meant to sit above this) and must preserve the
  `<answer>...</answer>` convention `rewards.py`'s `_extract_answer` depends on.
- `tests/unit/` is the only test tier — no live network, GPU, or dataset download in any test.
- `ruff check` and `ty check` must stay clean.

---

### Task 1: Prompt-building and row-formatting helpers

**Files:**
- Create: `src/turn_level_rewards/data.py`
- Test: `tests/unit/test_data.py`

**Interfaces:**
- Consumes: nothing from other modules.
- Produces: `_SYSTEM_PROMPT: str`, `_build_prompt(question: str) -> list[dict[str, str]]`,
  `_row_with_prompt(question: str, golden_answers: list[str], metadata: dict) -> dict`,
  `_format_train_row(row: dict) -> dict`, `_format_eval_row(row: dict) -> dict` — all consumed by
  Task 2's `load_train_dataset`/`load_eval_dataset`, in the same file.

- [ ] **Step 1: Write the failing tests for the row-formatting helpers**

Create `tests/unit/test_data.py`:

```python
from turn_level_rewards.data import _format_eval_row, _format_train_row

TRAIN_ROW = {
    "id": "train_0",
    "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
    "golden_answers": ["Arthur's Magazine"],
    "data_source": "hotpotqa",
    "prompt": [{"role": "user", "content": "old text-tag prompt, discarded"}],
    "ability": "fact-reasoning",
    "metadata": {
        "type": "comparison",
        "level": "medium",
        "supporting_facts": {"title": ["Arthur's Magazine", "First for Women"], "sent_id": [0, 0]},
        "context": {
            "title": ["Arthur's Magazine", "First for Women"],
            "sentences": [["Arthur's Magazine sentence."], ["First for Women sentence."]],
        },
    },
}

EVAL_ROW = {
    "id": "5a8b57f2",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    "context": {
        "title": ["Scott Derrickson", "Ed Wood"],
        "sentences": [["Scott Derrickson sentence."], ["Ed Wood sentence."]],
    },
}


def test_format_train_row_builds_system_then_user_prompt_with_question():
    row = _format_train_row(TRAIN_ROW)

    roles = [m["role"] for m in row["prompt"]]
    assert roles == ["system", "user"]
    assert row["prompt"][1]["content"] == TRAIN_ROW["question"]
    assert "at most 2 searches" in row["prompt"][0]["content"]


def test_format_train_row_passes_through_golden_answers_and_metadata():
    row = _format_train_row(TRAIN_ROW)

    assert row["golden_answers"] == ["Arthur's Magazine"]
    assert row["metadata"] == TRAIN_ROW["metadata"]


def test_format_train_row_drops_original_prompt_and_source_columns():
    row = _format_train_row(TRAIN_ROW)

    assert set(row.keys()) == {"prompt", "question", "golden_answers", "metadata"}


def test_format_eval_row_wraps_answer_into_golden_answers_list():
    row = _format_eval_row(EVAL_ROW)

    assert row["golden_answers"] == ["yes"]


def test_format_eval_row_nests_four_top_level_fields_under_metadata():
    row = _format_eval_row(EVAL_ROW)

    assert row["metadata"] == {
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
        "context": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sentences": [["Scott Derrickson sentence."], ["Ed Wood sentence."]],
        },
    }


def test_format_eval_row_builds_same_prompt_shape_as_train():
    row = _format_eval_row(EVAL_ROW)

    roles = [m["role"] for m in row["prompt"]]
    assert roles == ["system", "user"]
    assert row["prompt"][1]["content"] == EVAL_ROW["question"]


def test_train_and_eval_rows_have_identical_column_contract():
    train_row = _format_train_row(TRAIN_ROW)
    eval_row = _format_eval_row(EVAL_ROW)

    assert set(train_row.keys()) == set(eval_row.keys())
    assert set(train_row["metadata"].keys()) == set(eval_row["metadata"].keys())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.data'`.

- [ ] **Step 3: Implement the helpers in `src/turn_level_rewards/data.py`**

```python
"""Dataset loading for train/eval, reshaped to one shared column contract.

env.py/rewards.py never need to know which source dataset a row came from -- both loaders
produce identical prompt/question/golden_answers/metadata columns.
"""

_SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching Wikipedia when needed.\n\n"
    "You have access to a `search` tool that looks up Wikipedia passages for a query. Reason "
    "about what you need to find out, and call `search` with a focused query if you need more "
    "information (at most 2 searches). Read the results and reason further before deciding "
    "whether you need another search. Once you are confident in the answer, give it wrapped in "
    "<answer>...</answer> tags (e.g. <answer>Paris</answer>) and nothing else. If search results "
    "aren't helpful, rely on your own knowledge rather than searching repeatedly."
)


def _build_prompt(question: str) -> list[dict[str, str]]:
    """Build the system+user prompt that teaches native tool-calling for a question."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _row_with_prompt(question: str, golden_answers: list[str], metadata: dict) -> dict:
    """Assemble the shared output row shape used by both loaders."""
    return {
        "prompt": _build_prompt(question),
        "question": question,
        "golden_answers": golden_answers,
        "metadata": metadata,
    }


def _format_train_row(row: dict) -> dict:
    """Reshape a PeterJinGo/nq_hotpotqa_train row -- golden_answers/metadata already match."""
    return _row_with_prompt(row["question"], row["golden_answers"], row["metadata"])


def _format_eval_row(row: dict) -> dict:
    """Reshape a hotpotqa/hotpot_qa row -- wraps answer, nests 4 top-level fields under metadata."""
    metadata = {
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": row["supporting_facts"],
        "context": row["context"],
    }
    return _row_with_prompt(row["question"], [row["answer"]], metadata)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/turn_level_rewards/data.py tests/unit/test_data.py`

Run: `uv run ruff check src/turn_level_rewards/data.py tests/unit/test_data.py`
Expected: `All checks passed!`

Run: `uv run ty check src/turn_level_rewards/data.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/data.py tests/unit/test_data.py
git commit -m "Add row-formatting helpers for the shared train/eval column contract"
```

---

### Task 2: `load_train_dataset` and `load_eval_dataset`

**Files:**
- Modify: `src/turn_level_rewards/data.py`
- Modify: `tests/unit/test_data.py`

**Interfaces:**
- Consumes: Task 1's `_format_train_row(row: dict) -> dict`, `_format_eval_row(row: dict) -> dict`.
- Produces: `load_train_dataset(n: int | None, seed: int = 42, *, load_dataset_fn: Callable[..., Dataset] = datasets.load_dataset) -> Dataset`, `load_eval_dataset(n: int | None, seed: int = 42, *, load_dataset_fn: Callable[..., Dataset] = datasets.load_dataset) -> Dataset` — consumed by `train.py` (Phase 4).

- [ ] **Step 1: Add the failing tests for both loaders**

First, update the two import lines at the very top of `tests/unit/test_data.py` (ruff's isort rule
requires imports at the top of the file, not interspersed with test code) to:

```python
import datasets

from turn_level_rewards.data import (
    _format_eval_row,
    _format_train_row,
    load_eval_dataset,
    load_train_dataset,
)
```

Then append to the end of `tests/unit/test_data.py`:

```python
TRAIN_ROWS = [
    TRAIN_ROW,
    {
        "id": "train_1",
        "question": "What year was 127 Hours released?",
        "golden_answers": ["2010"],
        "data_source": "hotpotqa",
        "prompt": [{"role": "user", "content": "old text-tag prompt, discarded"}],
        "ability": "fact-reasoning",
        "metadata": {
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["127 Hours"], "sent_id": [0]},
            "context": {"title": ["127 Hours"], "sentences": [["A 2010 survival drama film."]]},
        },
    },
    {
        "id": "train_2",
        "question": "A natural-questions-sourced row that must be filtered out.",
        "golden_answers": ["irrelevant"],
        "data_source": "nq",
        "prompt": [{"role": "user", "content": "old text-tag prompt, discarded"}],
        "ability": "fact-reasoning",
        "metadata": {
            "type": "single",
            "level": "easy",
            "supporting_facts": {"title": [], "sent_id": []},
            "context": {"title": [], "sentences": []},
        },
    },
]

EVAL_ROWS = [
    EVAL_ROW,
    {
        "id": "5a90620b",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": ["Paris"], "sent_id": [0]},
        "context": {"title": ["Paris"], "sentences": [["Paris is the capital of France."]]},
    },
]


def _fake_loader(rows):
    def load_dataset_fn(*args, **kwargs):
        return datasets.Dataset.from_list(rows)

    return load_dataset_fn


def test_load_train_dataset_filters_to_hotpotqa_only():
    ds = load_train_dataset(None, load_dataset_fn=_fake_loader(TRAIN_ROWS))

    assert len(ds) == 2
    assert all(row["question"] != TRAIN_ROWS[2]["question"] for row in ds)


def test_load_train_dataset_n_selects_exactly_n_rows():
    ds = load_train_dataset(1, seed=0, load_dataset_fn=_fake_loader(TRAIN_ROWS))

    assert len(ds) == 1


def test_load_train_dataset_n_none_returns_all_filtered_rows():
    ds = load_train_dataset(None, load_dataset_fn=_fake_loader(TRAIN_ROWS))

    assert len(ds) == 2


def test_load_eval_dataset_wraps_answer_into_golden_answers_list():
    ds = load_eval_dataset(None, load_dataset_fn=_fake_loader(EVAL_ROWS))

    assert [row["golden_answers"] for row in ds] == [["yes"], ["Paris"]]


def test_load_eval_dataset_nests_supporting_facts_and_context_under_metadata():
    ds = load_eval_dataset(1, load_dataset_fn=_fake_loader(EVAL_ROWS))
    row = ds[0]

    assert row["metadata"]["supporting_facts"]["title"] == ["Scott Derrickson", "Ed Wood"]
    assert row["metadata"]["context"]["title"] == ["Scott Derrickson", "Ed Wood"]


def test_load_train_and_eval_datasets_have_identical_column_contract():
    train_ds = load_train_dataset(1, load_dataset_fn=_fake_loader(TRAIN_ROWS))
    eval_ds = load_eval_dataset(1, load_dataset_fn=_fake_loader(EVAL_ROWS))

    assert set(train_ds.column_names) == set(eval_ds.column_names)
    assert set(train_ds[0]["metadata"].keys()) == set(eval_ds[0]["metadata"].keys())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_train_dataset' from 'turn_level_rewards.data'`.

- [ ] **Step 3: Implement the loaders in `src/turn_level_rewards/data.py`**

Add these imports to the top of the file and the two functions at the end:

```python
from collections.abc import Callable

import datasets
from datasets import Dataset
```

```python
def load_train_dataset(
    n: int | None,
    seed: int = 42,
    *,
    load_dataset_fn: Callable[..., Dataset] = datasets.load_dataset,
) -> Dataset:
    """Load PeterJinGo/nq_hotpotqa_train, filtered to hotpotqa rows, reshaped to the shared contract.

    Args:
        n: Number of rows to select after shuffling, or None for all filtered rows.
        seed: Shuffle seed.
        load_dataset_fn: Injectable seam for the real datasets.load_dataset call -- tests pass a
            fake returning an in-memory Dataset.
    """
    ds = load_dataset_fn("PeterJinGo/nq_hotpotqa_train", "default", split="train")
    ds = ds.filter(lambda row: row["data_source"] == "hotpotqa")
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    return ds.map(_format_train_row, remove_columns=ds.column_names)


def load_eval_dataset(
    n: int | None,
    seed: int = 42,
    *,
    load_dataset_fn: Callable[..., Dataset] = datasets.load_dataset,
) -> Dataset:
    """Load hotpotqa/hotpot_qa (distractor, validation), reshaped to the shared contract.

    Args:
        n: Number of rows to select after shuffling, or None for all rows.
        seed: Shuffle seed.
        load_dataset_fn: Injectable seam for the real datasets.load_dataset call -- tests pass a
            fake returning an in-memory Dataset.
    """
    ds = load_dataset_fn("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    return ds.map(_format_eval_row, remove_columns=ds.column_names)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data.py -v`
Expected: 13 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/turn_level_rewards/data.py tests/unit/test_data.py`

Run: `uv run ruff check src/turn_level_rewards/data.py tests/unit/test_data.py`
Expected: `All checks passed!`

Run: `uv run ty check src/turn_level_rewards/data.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/data.py tests/unit/test_data.py
git commit -m "Add load_train_dataset/load_eval_dataset with injectable loader seam"
```

---

### Task 3: Validation script, manual real-data check, and Phase 3 sign-off

**Files:**
- Create: `scripts/verify_phase3.py`
- Modify: `docs/phase-3-data-pipeline.md` (task checkboxes + Handoff notes)
- Modify: `CLAUDE.md` (roadmap table Status column)

**Interfaces:**
- Consumes: nothing from `src/turn_level_rewards/` at import time — shells out to `uv run
  pytest`/`ruff`/`ty` as subprocesses, and inspects `data.py`'s source via `ast` to confirm the
  injectable-seam parameter exists, mirroring `verify_phase2.py`'s pattern.
- Produces: a `PASS`/`FAIL` exit-code gate. This is the last Phase 3 task — no later task depends
  on it within this plan.

- [ ] **Step 1: Create `scripts/verify_phase3.py`**

```python
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
```

- [ ] **Step 2: Run it and iterate until PASS**

Run: `uv run python scripts/verify_phase3.py`
Expected: `PASS: unit tests, ruff, and ty are clean, and both loaders have an injectable seam.`

If it prints `FAIL`, read exactly which check failed, fix the corresponding part of `data.py` from
Tasks 1-2, and re-run this same command. Do not proceed until it prints `PASS`.

- [ ] **Step 3: Commit the validation script**

```bash
git add scripts/verify_phase3.py
git commit -m "Add Phase 3 exit-criteria validation script"
```

- [ ] **Step 4: Run the one-off manual real-data check**

This confirms the exit criterion that needs the real (multi-GB) dataset download — not part of
`tests/unit/` or the script above, run once by hand:

```bash
uv run python -c "
from turn_level_rewards.data import load_train_dataset, load_eval_dataset

train = load_train_dataset(None)
eval_ds = load_eval_dataset(None)

print('train rows:', len(train))
print('eval rows:', len(eval_ds))
print('columns match:', set(train.column_names) == set(eval_ds.column_names))

n_facts = sum(len(row['metadata']['supporting_facts']['title']) for row in train)
print('avg supporting facts/row:', n_facts / len(train))
"
```

Expected: `train rows: 90447`, `eval rows: 7405`, `columns match: True`, `avg supporting facts/row: 2.0`
(matching CLAUDE.md's already-confirmed dataset facts). If any number differs, stop and
investigate before continuing — do not adjust the expected numbers to match unexpected output
without first checking whether the filtering logic in Task 2 is correct.

- [ ] **Step 5: Update `docs/phase-3-data-pipeline.md`'s task checkboxes and Handoff notes**

Check off every completed task's `- [ ]` to `- [x]` in the Tasks and Exit criteria sections.
Replace the Handoff notes section (currently `(not yet started)`) with:

```markdown
## Handoff notes

- **Column contract (final, confirmed identical for both loaders)**: `prompt`
  (`list[{"role", "content"}]`, system+user), `question` (`str`), `golden_answers` (`list[str]`),
  `metadata` (`dict` with `type`, `level`, `supporting_facts: {"title", "sent_id"}`,
  `context: {"title", "sentences"}`). `env.py`'s `reset(self, metadata, **kwargs)` and
  `rewards.py`'s `outcome_reward` consume `metadata`/`golden_answers` unmodified from either
  loader.
- **`load_dataset_fn` is an injectable keyword-only seam** (defaulting to the real
  `datasets.load_dataset`) on both `load_train_dataset`/`load_eval_dataset` -- a deviation from
  this doc's originally-sketched signature, added deliberately per CLAUDE.md's dependency-inversion
  principle (see `docs/superpowers/specs/2026-07-04-phase-3-data-pipeline-design.md`). Phase 4's
  `train.py` calls both with no `load_dataset_fn` argument, getting the real loader by default.
- **System prompt** lives in `data.py`'s `_SYSTEM_PROMPT` and states "at most 2 searches" verbatim
  -- Phase 4 must set `GRPOConfig(max_tool_calling_iterations=N)` with `N` above this (CLAUDE.md
  recommends 4).
- **Real row counts confirmed** via the manual check in Task 3: 90,447 train rows, 7,405 eval
  rows, identical column contract, 2.00 avg supporting facts/row -- matches CLAUDE.md's
  already-documented facts, no surprises found.
- **`scripts/verify_phase3.py`** is the exit-criteria gate -- re-run it after any future change to
  `data.py`.
```

- [ ] **Step 6: Update the roadmap table in `CLAUDE.md`**

Change the Phase 3 row from:

```markdown
| 3 | Data pipeline: `data.py` | `docs/phase-3-data-pipeline.md` | Not started |
```

to:

```markdown
| 3 | Data pipeline: `data.py` | `docs/phase-3-data-pipeline.md` | **Done** — `scripts/verify_phase3.py` passes; real row counts confirmed (90,447 train / 7,405 eval); see phase doc's Handoff notes for the injectable-loader-seam deviation and the exact system prompt location |
```

- [ ] **Step 7: Commit the documentation updates**

```bash
git add docs/phase-3-data-pipeline.md CLAUDE.md
git commit -m "Mark Phase 3 done; record handoff notes for Phase 4"
```

---

## Self-Review Notes

- **Spec coverage**: prompt-building/row-formatting helpers with the exact system-prompt text and
  system+user split (Task 1), both loaders with the injectable seam, filtering, shuffling, `n`
  selection, and `remove_columns` cleanup (Task 2), and the validation-script gate + manual
  real-data check + Handoff notes/roadmap update (Task 3) — every item in the design spec has a
  task.
- **Placeholder scan**: no TBD/TODO; every step has literal, complete code.
- **Type consistency**: `_format_train_row`/`_format_eval_row` signatures (`dict -> dict`) match
  how `load_train_dataset`/`load_eval_dataset` pass them to `.map()` in Task 2; `load_dataset_fn`'s
  type (`Callable[..., Dataset]`) matches every fake defined in `test_data.py`
  (`load_dataset_fn(*args, **kwargs) -> Dataset`); the shared `TRAIN_ROW`/`EVAL_ROW` fixtures
  defined in Task 1 are reused (not redefined) in Task 2's `TRAIN_ROWS`/`EVAL_ROWS` lists.
