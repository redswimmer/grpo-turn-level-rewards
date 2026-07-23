# Phase 8b: Full judge-augmented training runs + evaluation + comparison

## Goal

Run full training with the LLM-judge reward wired in (built in Phase 8), evaluate, and produce a
final comparison against Phase 7b's deterministic-reward `ppo`/`mt_ppo` results — closing the same
gap for the judge track that 7b closes for the deterministic track (see CLAUDE.md's Roadmap
section, 2026-07-23 addendum).

**This doc is intentionally a thin stub.** Its concrete task list depends on decisions and
measurements that don't exist yet: Phase 8's chosen judge-combination formula, the real observed
judge latency/cost per rollout, and the 20b-vs-120b judge-quality validation result. Flesh this
out for real once Phase 8 lands — don't treat the placeholder tasks below as final.

## Read first

Phase 8's Handoff notes (`docs/phase-8-llm-judge.md`) — the combination formula actually chosen,
real cost/latency numbers, and any Bedrock API surprises. Phase 7b's Handoff notes
(`docs/phase-7b-full-ppo-runs.md`) — the eval-path design and chart system built there should be
reused here, not rebuilt.

## Prerequisites (entry state)

- Phase 8 done: judge reward wired in, smoke-tested against the real Bedrock endpoint, per
  `docs/phase-8-llm-judge.md`'s exit criteria (including its cost-per-run estimate).

## Tasks (placeholder — confirm/replace once Phase 8's real numbers exist)

- [ ] Before committing to a full run: check Phase 8's recorded cost estimate against the actual
      budget for this run (500 rollout-collection steps, both conditions) — this is a real-money
      decision (Bedrock inference), not a free compute choice like the deterministic runs. Flag to
      the user for confirmation before spending it, per this repo's general "confirm before costly
      irreversible actions" norm.
- [ ] Full training runs, both judge-augmented conditions, mirroring 7b's structure.
- [ ] Held-out evaluation, reusing 7b's eval-path design.
- [ ] Comparison write-up: judge-augmented vs. deterministic-reward results, using 7b's matplotlib
      visual system for consistency (not a new chart style).

## Exit criteria (all must be true before handing off)

- [ ] Both judge-augmented conditions trained to completion within a confirmed budget.
- [ ] Held-out evaluation completed.
- [ ] Comparison against Phase 7b's deterministic results recorded, using the shared chart system.

## Handoff notes

<!-- Fill in after completing this phase. Leave this section for the next fresh agent to read
first. -->

(not yet started)
