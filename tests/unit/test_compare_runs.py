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
