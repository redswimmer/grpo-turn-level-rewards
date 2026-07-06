# Phase 6: evaluation + comparison — implementation design

## Context

`docs/phase-6-evaluation-comparison.md` already resolves the hard question for this phase
("how do you drive multi-turn generation for eval outside `GRPOTrainer`?" — you don't; build an
eval-only `GRPOTrainer` and call its standard `.evaluate()`) and has verified that approach
against a real checkpoint. It also already specifies exactly what `compare_runs.py` must plot and
the four criteria for deciding whether more training is needed. This spec covers what that phase
doc leaves as implementation detail: module architecture, file/directory layout, test scope, and
the execution plan for the real run.

Two decisions were made in this session that aren't in the phase doc:

1. **`results/` is a new top-level, git-committed directory** — not under `docs/` (that's
   prose/specs) and not under `outputs/` (that's gitignored wholesale, and mixing small final
   deliverables into a 53GB-and-growing gitignored tree risks losing them). Standard ML-repo
   convention: gitignore the big/regenerable training artifacts (checkpoints, logs), commit the
   small final deliverables (metrics summaries, comparison plots) in their own directory.
2. **README gets a folder-level "Project structure" section**, placed between "Roadmap" and
   "Reproducing this" — folders only, no file listing, ordered the standard ML-repo way
   (data → docs → outputs → results → scripts → src → tests). Already added in this session.

## Module architecture

### `src/turn_level_rewards/evaluate.py`

Mirrors `train.py`'s existing shape (pure builder / untested composition root / thin `main`) —
same split `train.py` already uses between `build_config` (unit-tested) and `build_trainer`
(integration surface, not unit-tested).

- `build_eval_config(condition, checkpoint_path, eval_batch_size) -> GRPOConfig` — pure. Sets
  `num_generations=2` (GRPO's hard minimum — advantage quality doesn't matter at eval time, only
  per-completion reward/metric values), `per_device_train_batch_size` and
  `per_device_eval_batch_size` both to `eval_batch_size`, `max_tool_calling_iterations=4` (same as
  training), `beta=0.0`, `max_completion_length=2048`, `report_to="none"`. Unit-tested the same
  way `test_train.py::test_build_config_*` tests `build_config`.
- `build_eval_trainer(condition, checkpoint_path, config) -> GRPOTrainer` — composition root, not
  unit-tested (same rationale as `build_trainer`: this is exactly the integration surface a real
  run validates). Constructs the real `GRPOTrainer` per the phase doc's verified snippet:
  `model=checkpoint_path`, `reward_funcs=get_reward_funcs(condition)`,
  `train_dataset=data.load_train_dataset(n=2, seed=42)` (unused filler — `.train()` never called),
  `eval_dataset=data.load_eval_dataset(n=None, seed=42)` (the full 7,405-row held-out set),
  `environment_factory=SearchEnv`.
- `main()` — CLI (`--condition`, `--checkpoint`, `--eval-batch-size`), calls
  `trainer.evaluate()`, writes the returned metrics dict as JSON to
  `results/{condition}_eval_metrics.json`.

### `scripts/compare_runs.py`

Follows the same DI split CLAUDE.md's guiding principles already establish for this repo: the
network/subprocess call is a seam, the transformation logic is pure and tested through that seam,
rendering is untested (like `build_trainer`).

- `fetch_trackio_metric(project, run, metric) -> list[dict]` — the seam. Wraps
  `uv run trackio get metric --project <p> --run <r> --metric <m> --json` via `subprocess.run`,
  parses the JSON, returns the `values` list. Injectable — tests pass a fake instead of shelling
  out.
- A pure data-prep function (e.g. `build_comparison_data(fetch_metric, eval_metrics_by_condition,
  run_names_by_condition) -> ComparisonData`) that calls the injected `fetch_metric` for the
  `train/exact_match`, `train/f1`, `train/tools/call_frequency` curves for both conditions'
  actual run names (`outcome_only-300steps-20260705-160524`,
  `turn_level-300steps-20260705-173317` — confirmed real and populated against trackio directly
  in this session) and combines them with the two loaded eval-metrics JSONs into one structure.
  Unit-tested with a fake `fetch_metric`.
- Plotting functions (matplotlib, untested) producing the four artifacts the phase doc specifies,
  written to `results/`:
  1. `em_f1_training_curves.png` — EM/F1 vs. step, both conditions overlaid (not raw composite
     reward — the phase doc is explicit that `turn_level`'s reward isn't on the same scale as
     `outcome_only`'s).
  2. `final_em_f1_comparison.png` — final held-out EM/F1 bars.
  3. `final_retrieval_rate.png` — final held-out retrieval-rate bar (`turn_level` only).
  4. `tool_call_frequency.png` — `outcome_only` vs. `turn_level` `train/tools/call_frequency`
     curves side by side — this is the direct check of the paper's claimed mechanism
     ("`GRPO-OR` gradually stops calling search tools").
- `main()` — CLI wiring: run names, eval-json paths, output dir (defaults to `results/`).

### Dependencies

Add `matplotlib` to `pyproject.toml`. Nothing else needed — no `pandas`; trackio's JSON output is
already plain step/value pairs, and the data-prep function works with plain lists/dicts per
CLAUDE.md's "keep abstractions thin" principle.

## Execution plan (this session runs it to completion, per user's choice)

1. Add `matplotlib` dependency; create `results/`.
2. Implement `evaluate.py` + `test_evaluate.py`; run tests.
3. Canary: run `evaluate.py` at a small held-out row count across a couple of
   `--eval-batch-size` values (the phase doc suggests starting around 32) to find a safe, fast
   batch size and get a real per-row wall-clock estimate — don't assume either, per Phase 5's own
   canary discipline.
4. Launch the real full evaluation (7,405 rows) for `outcome_only/checkpoint-300`, then
   `turn_level/checkpoint-300`, via `systemd-run --user --scope` (Phase 5's fix for the transient
   cgroup-kill issue on long-running processes in this environment).
5. Implement `compare_runs.py` + `test_compare_runs.py`; run tests.
6. Run `compare_runs.py` against the real trackio data + the two real eval-metrics JSONs to
   produce the four plots in `results/`.
7. Check the four "more training needed?" criteria in `docs/phase-6-evaluation-comparison.md`
   against the real numbers; record the outcome either way.
8. Update README's Results section (real held-out numbers, not just training-batch ones) and
   Roadmap bullet (GRPO comparison → done); update CLAUDE.md's roadmap table row 6; fill in
   `docs/phase-6-evaluation-comparison.md`'s Handoff notes.

## Testing scope

Consistent with CLAUDE.md's existing test-pyramid principle (fast, deterministic, no GPU/network,
one tier only — `tests/unit/`):

- `test_evaluate.py`: `build_eval_config`'s fields only (batch-size wiring, condition-invariant
  fields, `report_to="none"`) — no real `GRPOTrainer`.
- `test_compare_runs.py`: the pure data-prep function via a fake `fetch_trackio_metric` — no real
  subprocess call, no real matplotlib rendering.
- No new test tier. The real evaluation run (step 4 above) is this phase's live smoke test,
  analogous to Phase 4's — not something `tests/unit/` attempts to cover.
