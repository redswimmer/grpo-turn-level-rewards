# Phase 6 Evaluation + Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evaluate.py` (runs a trained checkpoint over the held-out set via
`GRPOTrainer.evaluate()`) and `scripts/compare_runs.py` (pulls trackio training curves + the two
eval-metrics JSONs, produces the four comparison plots), plus a static exit-criteria script — the
three code deliverables Phase 6 needs before the real evaluation runs can happen.

**Architecture:** Two new CLI modules, each following the pure-builder/untested-composition-root
split `train.py` already establishes (`build_config`/`build_trainer` → `build_eval_config`/
`build_eval_trainer` for evaluate.py; a similar seam/pure-function split for compare_runs.py's
trackio fetch vs. its data-prep function). One new top-level package (`scripts/`, previously just
loose scripts) so its pure logic is unit-testable. No changes to existing modules.

**Tech Stack:** Python 3.13, `trl.GRPOTrainer`/`GRPOConfig`, `matplotlib` (new dependency),
`pytest`, `ruff`, `ty`.

## Global Constraints

- Test location: `tests/unit/` only — no new test tiers (CLAUDE.md's "Repo layout" section).
- `results/` is a new top-level, **git-committed** directory (not gitignored, unlike `outputs/`)
  — holds `{condition}_eval_metrics.json` and the four comparison PNGs. Nothing in this plan
  writes there yet (that happens after this plan, against real data) — the code just needs to
  create the directory on demand and write there.
- `evaluate.py`'s eval-only `GRPOTrainer` construction must match
  `docs/phase-6-evaluation-comparison.md`'s already-verified snippet: `num_generations=2`,
  `report_to="none"`, `max_tool_calling_iterations=4`, `beta=0.0`, `max_completion_length=2048`,
  `train_dataset=data.load_train_dataset(n=2, seed=42)` (unused filler),
  `eval_dataset=data.load_eval_dataset(n=<eval_size>, seed=42)`, `environment_factory=SearchEnv`.
- Confirmed real TRL behavior (checked directly against the installed version, not assumed):
  `GRPOConfig(report_to="none")` produces `report_to == []`, not `["none"]`.
- Confirmed real trackio metric names (queried directly against this repo's own trackio data):
  `train/exact_match`, `train/f1`, `train/tools/call_frequency`. Real run names from Phase 5:
  `outcome_only-300steps-20260705-160524`, `turn_level-300steps-20260705-173317` (both
  `checkpoint-300`).
- `scripts/` needs to become an importable package for `test_compare_runs.py` to import it.
  Confirmed working combination (tested directly in this session): an empty `scripts/__init__.py`
  plus `[tool.pytest.ini_options]\npythonpath = ["."]` in `pyproject.toml`. Without the
  `pythonpath` entry, `import scripts` raises `ModuleNotFoundError` even with `__init__.py`
  present, because `scripts/` isn't part of the installed package (only `src/turn_level_rewards`
  is, per `[tool.hatch.build.targets.wheel]`).
- Canary/full-run commands (used after this plan, not in it):
  - Canary: `uv run python -m turn_level_rewards.evaluate --condition outcome_only --checkpoint outputs/outcome_only/checkpoint-300 --eval-size 32 --eval-batch-size 8` (then try larger `--eval-batch-size` values).
  - Full run: `--eval-size 7405 --eval-batch-size <picked from canary>`, once per condition.

---

### Task 1: `evaluate.py`

**Files:**
- Create: `src/turn_level_rewards/evaluate.py`
- Create: `tests/unit/test_evaluate.py`

**Interfaces:**
- Consumes: `turn_level_rewards.data.load_train_dataset`/`load_eval_dataset`,
  `turn_level_rewards.rewards.get_reward_funcs`, `turn_level_rewards.env.SearchEnv` (all existing,
  unmodified).
- Produces: `Condition` (type alias, same values as `train.py`'s), `build_eval_config(condition:
  Condition, eval_batch_size: int) -> GRPOConfig`, `build_eval_trainer(condition: Condition,
  checkpoint: str, eval_size: int | None, config: GRPOConfig) -> GRPOTrainer`, `main() -> None`.
  Task 3's `verify_phase6.py` imports `build_eval_config` directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_evaluate.py`:

```python
"""Fast, GPU-free tests for evaluate.py's build_eval_config and CLI parsing.

No real GRPOTrainer, model, or checkpoint is constructed here -- that integration surface is
what the real canary/full evaluation run (not tests/unit/) covers instead, per CLAUDE.md's
Guiding principles.
"""

import pytest
from turn_level_rewards.evaluate import Condition, _parse_args, build_eval_config


def _build(condition: Condition, eval_batch_size: int = 8) -> object:
    return build_eval_config(condition=condition, eval_batch_size=eval_batch_size)


def test_build_eval_config_fixed_fields_identical_across_conditions():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    for config in (outcome_config, turn_config):
        assert config.num_generations == 2
        assert config.max_tool_calling_iterations == 4
        assert config.beta == 0.0
        assert config.max_completion_length == 2048
        assert config.report_to == []


def test_build_eval_config_batch_size_wiring():
    config = _build("outcome_only", eval_batch_size=8)

    assert config.per_device_train_batch_size == 8
    assert config.per_device_eval_batch_size == 8
    assert config.generation_batch_size == 8
    assert config.generation_batch_size % config.num_generations == 0


def test_build_eval_config_output_dir_differs_by_condition():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    assert outcome_config.output_dir == "outputs/outcome_only/eval-scratch"
    assert turn_config.output_dir == "outputs/turn_level/eval-scratch"


def test_build_eval_config_rejects_odd_eval_batch_size():
    with pytest.raises(ValueError, match="even"):
        build_eval_config(condition="outcome_only", eval_batch_size=3)


def test_parse_args_defaults():
    args = _parse_args(
        ["--condition", "outcome_only", "--checkpoint", "outputs/outcome_only/checkpoint-300"]
    )

    assert args.condition == "outcome_only"
    assert args.checkpoint == "outputs/outcome_only/checkpoint-300"
    assert args.eval_batch_size == 2
    assert args.eval_size == 4
    assert args.output is None


def test_parse_args_condition_required():
    with pytest.raises(SystemExit):
        _parse_args(["--checkpoint", "outputs/outcome_only/checkpoint-300"])


def test_parse_args_checkpoint_required():
    with pytest.raises(SystemExit):
        _parse_args(["--condition", "outcome_only"])


def test_parse_args_condition_choices_enforced():
    with pytest.raises(SystemExit):
        _parse_args(["--condition", "bogus", "--checkpoint", "x"])


def test_parse_args_overrides():
    args = _parse_args(
        [
            "--condition",
            "turn_level",
            "--checkpoint",
            "outputs/turn_level/checkpoint-300",
            "--eval-batch-size",
            "32",
            "--eval-size",
            "7405",
            "--output",
            "results/turn_level_eval_metrics.json",
        ]
    )

    assert args.condition == "turn_level"
    assert args.checkpoint == "outputs/turn_level/checkpoint-300"
    assert args.eval_batch_size == 32
    assert args.eval_size == 7405
    assert args.output == "results/turn_level_eval_metrics.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.evaluate'`.

- [ ] **Step 3: Write `evaluate.py`**

Create `src/turn_level_rewards/evaluate.py`:

```python
"""evaluate.py: run a trained checkpoint over the held-out set via GRPOTrainer.evaluate().

See docs/phase-6-evaluation-comparison.md's "evaluate.py's design" section for why this
approach -- construct a GRPOTrainer for evaluation only and call its standard .evaluate() --
is correct: GRPOTrainer already overrides prediction_step to run the exact same
generation-and-score path training uses, confirmed there against a real checkpoint.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from trl import GRPOConfig, GRPOTrainer

from turn_level_rewards import data
from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import get_reward_funcs

Condition = Literal["outcome_only", "turn_level"]


def build_eval_config(condition: Condition, eval_batch_size: int) -> GRPOConfig:
    """Build the GRPOConfig for an eval-only GRPOTrainer.

    num_generations is fixed at 2 -- GRPO's hard minimum (>=2, needs group variance for
    advantages); we don't care about advantage quality at eval time, only the per-completion
    reward/metric values GRPOTrainer.evaluate() returns, so the minimum is the cheapest valid
    choice. per_device_train_batch_size is set to eval_batch_size too, even though .train() is
    never called: GRPOConfig enforces the same generation_batch_size % num_generations == 0
    divisibility constraint training does, and per_device_eval_batch_size alone doesn't satisfy
    it. eval_batch_size must therefore be even (a multiple of num_generations=2) -- checked
    explicitly below so a bad CLI value fails fast with a clear message instead of a cryptic
    error deep inside GRPOTrainer's batching logic.
    """
    if eval_batch_size % 2 != 0:
        raise ValueError(
            f"eval_batch_size must be even (num_generations=2); got {eval_batch_size}"
        )
    return GRPOConfig(
        output_dir=f"outputs/{condition}/eval-scratch",
        num_generations=2,
        per_device_train_batch_size=eval_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        max_tool_calling_iterations=4,
        beta=0.0,
        max_completion_length=2048,
        report_to="none",
    )


def build_eval_trainer(
    condition: Condition, checkpoint: str, eval_size: int | None, config: GRPOConfig
) -> GRPOTrainer:
    """Composition root: real model checkpoint, real SearchEnv, real held-out data.

    Not unit-tested -- this is exactly the integration surface a real canary/full run validates,
    matching train.py's build_trainer.
    """
    return GRPOTrainer(
        model=checkpoint,
        reward_funcs=get_reward_funcs(condition),
        args=config,
        train_dataset=data.load_train_dataset(n=2, seed=42),  # unused filler; .train() never called
        eval_dataset=data.load_eval_dataset(n=eval_size, seed=42),
        environment_factory=SearchEnv,  # ty: ignore[invalid-argument-type]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluate.py's CLI arguments.

    The bare invocation (just --condition/--checkpoint) evaluates a tiny 4-row slice at a small
    batch size -- a smoke-scale default mirroring train.py's. The real full held-out run passes
    --eval-size 7405 explicitly, the same "pass the exact full size" convention train.py's Phase
    5 launch already established (--train-size 90447).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the held-out set (see CLAUDE.md)."
    )
    parser.add_argument("--condition", required=True, choices=["outcome_only", "turn_level"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    config = build_eval_config(args.condition, args.eval_batch_size)
    trainer = build_eval_trainer(args.condition, args.checkpoint, args.eval_size, config)
    metrics = trainer.evaluate()

    output_path = Path(args.output or f"results/{args.condition}_eval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote eval metrics to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluate.py -v`
Expected: PASS, all 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/evaluate.py tests/unit/test_evaluate.py
git commit -m "Add evaluate.py: run a checkpoint over the held-out set via GRPOTrainer.evaluate()"
```

---

### Task 2: `scripts/compare_runs.py`

**Files:**
- Modify: `pyproject.toml` (add `matplotlib` to `dependencies`; add a new
  `[tool.pytest.ini_options]` table with `pythonpath = ["."]`)
- Create: `scripts/__init__.py` (empty — makes `scripts` an importable package)
- Create: `scripts/compare_runs.py`
- Create: `tests/unit/test_compare_runs.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `fetch_trackio_metric(project: str, run: str, metric: str) -> list[dict]`,
  `load_eval_metrics(path: str) -> dict`, `build_comparison_data(fetch_metric, run_names:
  dict[str, str], eval_metrics: dict[str, dict], project: str = PROJECT) -> dict`,
  `plot_em_f1_training_curves(data, out_path)`, `plot_final_em_f1_comparison(data, out_path)`,
  `plot_final_retrieval_rate(data, out_path)`, `plot_tool_call_frequency(data, out_path)`,
  `_parse_args(argv) -> Namespace`, `main(argv: list[str] | None = None) -> None`.

- [ ] **Step 1: Dependency and import plumbing**

In `pyproject.toml`, add `"matplotlib>=3.10.0",` to the `dependencies` list (alphabetically,
after `"jmespath>=1.0.1",` and before `"pydantic>=2.13.4",`), and append at the end of the file:

```toml

[tool.pytest.ini_options]
pythonpath = ["."]
```

Create empty `scripts/__init__.py` (touch it — no content needed).

Run: `uv sync`
Expected: `matplotlib` installed, no errors.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_compare_runs.py`:

```python
"""Fast, GPU-free tests for compare_runs.py's data-prep function, CLI parsing, and end-to-end
plot-file production (via a fake fetch_trackio_metric -- no real subprocess/trackio call).
"""

import json

import pytest
from scripts.compare_runs import _parse_args, build_comparison_data, main


def _fake_fetch(values_by_run_metric):
    def fetch(project, run, metric):
        return [{"step": s, "value": v} for s, v in values_by_run_metric[(run, metric)]]

    return fetch


def test_build_comparison_data_shapes_training_curves_per_condition():
    fetch = _fake_fetch(
        {
            ("run-a", "train/exact_match"): [(1, 0.1), (2, 0.2)],
            ("run-a", "train/f1"): [(1, 0.2), (2, 0.3)],
            ("run-a", "train/tools/call_frequency"): [(1, 1.0), (2, 0.5)],
            ("run-b", "train/exact_match"): [(1, 0.15)],
            ("run-b", "train/f1"): [(1, 0.25)],
            ("run-b", "train/tools/call_frequency"): [(1, 1.2)],
        }
    )
    run_names = {"outcome_only": "run-a", "turn_level": "run-b"}
    eval_metrics = {
        "outcome_only": {"eval_exact_match": 0.2},
        "turn_level": {"eval_exact_match": 0.3},
    }

    data = build_comparison_data(fetch, run_names, eval_metrics)

    assert data["training_curves"]["outcome_only"]["train/exact_match"] == [(1, 0.1), (2, 0.2)]
    assert data["training_curves"]["turn_level"]["train/f1"] == [(1, 0.25)]
    assert data["training_curves"]["outcome_only"]["train/tools/call_frequency"] == [
        (1, 1.0),
        (2, 0.5),
    ]
    assert data["eval_metrics"] == eval_metrics


def test_build_comparison_data_passes_project_through_to_fetch_metric():
    seen_projects = []

    def fetch(project, run, metric):
        seen_projects.append(project)
        return []

    build_comparison_data(
        fetch,
        {"outcome_only": "run-a", "turn_level": "run-b"},
        {"outcome_only": {}, "turn_level": {}},
        project="custom-project",
    )

    assert seen_projects and all(p == "custom-project" for p in seen_projects)


def test_parse_args_defaults():
    args = _parse_args(["--outcome-run", "run-a", "--turn-run", "run-b"])

    assert args.outcome_run == "run-a"
    assert args.turn_run == "run-b"
    assert args.outcome_eval_json == "results/outcome_only_eval_metrics.json"
    assert args.turn_eval_json == "results/turn_level_eval_metrics.json"
    assert args.output_dir == "results"


def test_parse_args_outcome_run_required():
    with pytest.raises(SystemExit):
        _parse_args(["--turn-run", "run-b"])


def test_main_writes_four_plot_files(tmp_path, monkeypatch):
    outcome_json = tmp_path / "outcome.json"
    turn_json = tmp_path / "turn.json"
    outcome_json.write_text(json.dumps({"eval_exact_match": 0.2, "eval_f1": 0.3}))
    turn_json.write_text(
        json.dumps({"eval_exact_match": 0.25, "eval_f1": 0.35, "eval_retrieval_fraction": 0.4})
    )

    fake_values = {
        ("outcome-run", "train/exact_match"): [(1, 0.1)],
        ("outcome-run", "train/f1"): [(1, 0.2)],
        ("outcome-run", "train/tools/call_frequency"): [(1, 1.0)],
        ("turn-run", "train/exact_match"): [(1, 0.15)],
        ("turn-run", "train/f1"): [(1, 0.25)],
        ("turn-run", "train/tools/call_frequency"): [(1, 1.2)],
    }

    def fake_fetch(project, run, metric):
        return [{"step": s, "value": v} for s, v in fake_values[(run, metric)]]

    monkeypatch.setattr("scripts.compare_runs.fetch_trackio_metric", fake_fetch)

    out_dir = tmp_path / "out"
    main(
        [
            "--outcome-run",
            "outcome-run",
            "--turn-run",
            "turn-run",
            "--outcome-eval-json",
            str(outcome_json),
            "--turn-eval-json",
            str(turn_json),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert (out_dir / "em_f1_training_curves.png").exists()
    assert (out_dir / "final_em_f1_comparison.png").exists()
    assert (out_dir / "final_retrieval_rate.png").exists()
    assert (out_dir / "tool_call_frequency.png").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_compare_runs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.compare_runs'`.

- [ ] **Step 4: Write `scripts/compare_runs.py`**

Create `scripts/compare_runs.py`:

```python
"""compare_runs.py: pull trackio training curves + evaluate.py's held-out metrics, produce the
four comparison plots docs/phase-6-evaluation-comparison.md specifies.
"""

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = "turn-level-rewards"
TRAINING_METRICS = ("train/exact_match", "train/f1", "train/tools/call_frequency")

FetchMetric = Callable[[str, str, str], list[dict[str, Any]]]


def fetch_trackio_metric(project: str, run: str, metric: str) -> list[dict[str, Any]]:
    """Fetch one metric's logged (step, value) series for one trackio run.

    The seam: wraps `trackio get metric --json`. Tests inject a fake instead of shelling out.
    """
    result = subprocess.run(
        [
            "uv",
            "run",
            "trackio",
            "get",
            "metric",
            "--project",
            project,
            "--run",
            run,
            "--metric",
            metric,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["values"]


def load_eval_metrics(path: str) -> dict[str, Any]:
    """Read one condition's evaluate.py output json."""
    return json.loads(Path(path).read_text())


def build_comparison_data(
    fetch_metric: FetchMetric,
    run_names: dict[str, str],
    eval_metrics: dict[str, dict[str, Any]],
    project: str = PROJECT,
) -> dict[str, Any]:
    """Combine fetched training curves + loaded eval metrics into one plottable structure.

    Args:
        fetch_metric: injected trackio-fetch seam (see fetch_trackio_metric).
        run_names: {"outcome_only": <trackio run name>, "turn_level": <trackio run name>}.
        eval_metrics: {"outcome_only": <evaluate.py output dict>, "turn_level": <...>}.
        project: trackio project name.

    Returns:
        {"training_curves": {condition: {metric: [(step, value), ...]}},
         "eval_metrics": eval_metrics}
    """
    training_curves = {}
    for condition, run in run_names.items():
        training_curves[condition] = {
            metric: [(v["step"], v["value"]) for v in fetch_metric(project, run, metric)]
            for metric in TRAINING_METRICS
        }
    return {"training_curves": training_curves, "eval_metrics": eval_metrics}


def plot_em_f1_training_curves(data: dict[str, Any], out_path: Path) -> None:
    """Plot 1: EM/F1 vs. step, both conditions overlaid -- not raw composite reward, since
    turn_level's reward includes the extra turn_reward term and isn't on the same scale as
    outcome_only's (see docs/phase-6-evaluation-comparison.md).
    """
    fig, (em_ax, f1_ax) = plt.subplots(1, 2, figsize=(10, 4))
    for condition, curves in data["training_curves"].items():
        em_points = curves["train/exact_match"]
        f1_points = curves["train/f1"]
        em_ax.plot([p[0] for p in em_points], [p[1] for p in em_points], label=condition)
        f1_ax.plot([p[0] for p in f1_points], [p[1] for p in f1_points], label=condition)
    em_ax.set_title("Exact match")
    f1_ax.set_title("F1")
    for ax in (em_ax, f1_ax):
        ax.set_xlabel("step")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_final_em_f1_comparison(data: dict[str, Any], out_path: Path) -> None:
    """Plot 2: final held-out EM/F1 bars, one pair of bars per condition."""
    conditions = list(data["eval_metrics"].keys())
    em_values = [data["eval_metrics"][c]["eval_exact_match"] for c in conditions]
    f1_values = [data["eval_metrics"][c]["eval_f1"] for c in conditions]

    positions = range(len(conditions))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar([p - width / 2 for p in positions], em_values, width, label="Exact match")
    ax.bar([p + width / 2 for p in positions], f1_values, width, label="F1")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Held-out score")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_final_retrieval_rate(data: dict[str, Any], out_path: Path) -> None:
    """Plot 3: final held-out retrieval-rate bar -- only conditions whose eval metrics include
    eval_retrieval_fraction (outcome_only's reward functions never compute it).
    """
    conditions = [c for c, m in data["eval_metrics"].items() if "eval_retrieval_fraction" in m]
    values = [data["eval_metrics"][c]["eval_retrieval_fraction"] for c in conditions]

    fig, ax = plt.subplots()
    ax.bar(conditions, values)
    ax.set_ylabel("Held-out retrieval_fraction")
    fig.savefig(out_path)
    plt.close(fig)


def plot_tool_call_frequency(data: dict[str, Any], out_path: Path) -> None:
    """Plot 4: outcome_only vs. turn_level train/tools/call_frequency curves -- the direct check
    of the paper's claimed mechanism ("GRPO-OR gradually stops calling search tools").
    """
    fig, ax = plt.subplots()
    for condition, curves in data["training_curves"].items():
        points = curves["train/tools/call_frequency"]
        ax.plot([p[0] for p in points], [p[1] for p in points], label=condition)
    ax.set_xlabel("step")
    ax.set_ylabel("tool calls per completion")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare outcome_only vs turn_level trackio curves + held-out eval metrics."
    )
    parser.add_argument("--outcome-run", required=True)
    parser.add_argument("--turn-run", required=True)
    parser.add_argument("--outcome-eval-json", default="results/outcome_only_eval_metrics.json")
    parser.add_argument("--turn-eval-json", default="results/turn_level_eval_metrics.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--project", default=PROJECT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_names = {"outcome_only": args.outcome_run, "turn_level": args.turn_run}
    eval_metrics = {
        "outcome_only": load_eval_metrics(args.outcome_eval_json),
        "turn_level": load_eval_metrics(args.turn_eval_json),
    }
    data = build_comparison_data(
        fetch_trackio_metric, run_names, eval_metrics, project=args.project
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_em_f1_training_curves(data, output_dir / "em_f1_training_curves.png")
    plot_final_em_f1_comparison(data, output_dir / "final_em_f1_comparison.png")
    plot_final_retrieval_rate(data, output_dir / "final_retrieval_rate.png")
    plot_tool_call_frequency(data, output_dir / "tool_call_frequency.png")
    print(f"Wrote 4 comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_compare_runs.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock scripts/__init__.py scripts/compare_runs.py tests/unit/test_compare_runs.py
git commit -m "Add compare_runs.py: trackio curves + held-out metrics -> 4 comparison plots"
```

---

### Task 3: Phase 6 exit-criteria script (static portion)

**Files:**
- Create: `scripts/verify_phase6.py`

**Interfaces:**
- Consumes: `turn_level_rewards.evaluate.build_eval_config` (Task 1).
- Produces: a script exiting `0` with a `PASS` message, or `1` with a `FAIL` message listing
  every failing check — same contract as `scripts/verify_phase5.py`.

- [ ] **Step 1: Write the script**

Create `scripts/verify_phase6.py`:

```python
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
                    f"build_eval_config({condition!r}).{field} == {actual!r}, "
                    f"expected {expected!r}"
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
```

- [ ] **Step 2: Run the script to verify it passes**

Run: `uv run python scripts/verify_phase6.py`
Expected: exits 0, prints the `PASS` message above.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_phase6.py
git commit -m "Add Phase 6 exit-criteria validation script (static portion)"
```

---

## After this plan: not part of any task above

Once all 3 tasks are reviewed and merged, the real GPU/network work happens manually (not
delegated to a subagent), per `docs/phase-6-evaluation-comparison.md`'s Tasks and Decision
sections:

1. **Canary**: `uv run python -m turn_level_rewards.evaluate --condition outcome_only --checkpoint
   outputs/outcome_only/checkpoint-300 --eval-size 32 --eval-batch-size 8`, watching wall-clock
   directly; try larger `--eval-batch-size` values (e.g. 16, 32) until one is both fast and
   OOM-free, per Phase 5's own canary discipline.
2. **Full evaluation, `outcome_only`**: same command with `--eval-size 7405` and the picked batch
   size, via `systemd-run --user --scope` (Phase 5's fix for the transient-cgroup-kill issue on
   long-running processes in this environment) — writes `results/outcome_only_eval_metrics.json`.
3. **Full evaluation, `turn_level`**: same, for `outputs/turn_level/checkpoint-300` — writes
   `results/turn_level_eval_metrics.json`.
4. **Run `compare_runs.py`** with the real Phase 5 run names (`--outcome-run
   outcome_only-300steps-20260705-160524 --turn-run turn_level-300steps-20260705-173317`) — writes
   the four PNGs into `results/`.
5. **Check the four "more training needed?" criteria** in
   `docs/phase-6-evaluation-comparison.md` against the real numbers; record the outcome either way
   in that doc's Handoff notes.
6. **Update** README's Results section (real held-out numbers, referencing the four PNGs) and
   Roadmap bullet (GRPO comparison → done); update CLAUDE.md's roadmap table row 6; fill in
   `docs/phase-6-evaluation-comparison.md`'s Handoff notes and flip its Status to Done.
