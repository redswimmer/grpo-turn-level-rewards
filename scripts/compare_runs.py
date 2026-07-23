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
