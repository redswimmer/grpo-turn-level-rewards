# Phase 6: Evaluation + comparison

## Goal

Evaluate both trained checkpoints on the held-out set and produce the final comparison — the
actual answer to "does turn-level reward help, and how, in this simplified recreation."

## Read first

`CLAUDE.md` — "Goal" (recall the exact axes the paper itself compares on: accuracy,
stability/convergence speed, and the retrieval-rate mechanism believed to explain it) and
"Experiment tracking (trackio)". Read Phase 5's Handoff notes for checkpoint paths and anything
unusual already spotted in the training curves.

## Prerequisites (entry state)

- Phase 5 done: both checkpoints exist and are loadable; both runs' full metrics are in trackio.

## Tasks

- [ ] `src/turn_level_rewards/evaluate.py` — for each checkpoint: load it, run it over the held
      -out eval set (`data.load_eval_dataset`, reusing the real `SearchEnv`/retrieval server, no
      `num_generations>1` needed at eval time), compute and write to a metrics json: EM/F1
      (reusing `metrics.py` — same code the training-time `outcome_reward` used, so eval and
      training reward are provably consistent), supporting-fact retrieval rate
      (`environment.retrieval_fraction` averaged over the eval set), and format-compliance rate.
- [ ] `scripts/compare_runs.py` — pull each condition's trackio metrics
      (`trackio get metric --project turn-level-rewards --run <run> --metric <name> --json`) plus
      both `evaluate.py` output jsons; produce: (1) overlaid reward-vs-step curves for both
      conditions (stability/convergence-speed comparison), (2) final EM/F1 bar comparison, (3)
      final retrieval-rate bar comparison.
- [ ] Write up findings in plain language: did `turn_level` show faster/more stable convergence
      and higher final accuracy than `outcome_only`, as the paper claims? If not, is there a
      plausible reason given what's already documented in CLAUDE.md (e.g. the ~80% retrieval
      ceiling, the small training scale relative to the paper, hyperparameter choices)?

## Exit criteria (all must be true before handing off — this is the last phase)

- [ ] Both checkpoints evaluated on the same fixed held-out set.
- [ ] Comparison plots/tables produced and saved somewhere durable (not just a notebook cell).
- [ ] A short written summary of findings exists, explicitly stating whether this simplified
      recreation reproduces the paper's core claim or not, and why.

## Handoff notes

<!-- This is the final phase — record final results and any recommended follow-up experiments
here (e.g. scaling up train size/steps, trying the closed-corpus or pooled-BM25 alternatives from
CLAUDE.md's rejected-options table as a fidelity check, tuning turn_reward's magnitude). -->

(not yet started)
