# Phase 2: Core library — env, rewards, metrics + unit tests

## Goal

Implement `env.py`, `rewards.py`, `metrics.py` with dependency-injected seams (per CLAUDE.md's
"Guiding principles for code, tests, and dependencies"), plus a fast `tests/unit/` suite that
needs no GPU, no network, and no live retrieval server.

## Read first

`CLAUDE.md` — especially "TRL mechanics being relied on", "Reward design (the crux decision)",
and "Guiding principles for code, tests, and dependencies". This doc only covers this phase's
concrete tasks.

## Prerequisites (entry state)

- Phase 1's retrieval server **contract** must be finalized (request/response JSON shape) — this
  phase only needs the documented contract, not a live running server, since tests inject a fake
  retriever at that seam. If Phase 1 isn't fully done yet but the contract in CLAUDE.md is
  considered stable, this phase can start in parallel; check `docs/phase-1-retrieval-infra.md`'s
  Handoff notes for any contract changes discovered during that phase before assuming the
  contract in CLAUDE.md is exactly right.

## Tasks

- [ ] `src/turn_level_rewards/metrics.py`: `normalize_answer`, `exact_match`, `f1_score`
      (SQuAD-style, stdlib only — no new dependencies for this file).
- [ ] `src/turn_level_rewards/env.py`: `SearchEnv` class.
      - `reset(self, context, supporting_facts, **kwargs) -> str | None` — resets all mutable
        per-episode state (remember: TRL reuses instances from a pool; leftover state from a
        prior episode is a real bug class here, not a hypothetical).
      - `search(self, query: str) -> str` tool method — calls the retrieval server. **The HTTP
        client/call itself must be injectable** (constructor parameter, module-level default
        factory, or similar) — this is the seam principle 1 in CLAUDE.md refers to concretely.
        Extracts `{title, text}` from each returned document via
        `contents.split("\n")[0].strip('"')` (matching Search-R1's own parsing, confirmed in
        CLAUDE.md).
      - Tracks `retrieval_fraction` (dedup'd fraction of gold `supporting_facts` titles actually
        surfaced), capped at 1.0.
- [ ] `src/turn_level_rewards/rewards.py`:
      - `format_reward(completions, **kwargs)`
      - `outcome_reward(completions, golden_answers, **kwargs)` — SQuAD F1 + EM bonus, maxed over
        the `golden_answers` list (it's a list, not a single string — see CLAUDE.md's Dataset
        section).
      - `turn_reward(environments, **kwargs)` — `0.4 * environment.retrieval_fraction`.
      - `get_reward_funcs(condition: Literal["outcome_only", "turn_level"])` — returns the right
        list per CLAUDE.md's Reward design section.
      - All functions operate on plain data (strings/dicts/duck-typed objects) — no real
        `GRPOTrainer` or loaded model needed to exercise them.
- [ ] `tests/unit/test_metrics.py` — EM/F1 sanity pairs (identical strings, partial overlap,
      case/punctuation/article normalization, fully disjoint answers).
- [ ] `tests/unit/test_env.py` — inject a **fake retriever** (plain function/dict returning
      canned documents, no real HTTP). Cover: a query that should hit a gold title updates
      `retrieval_fraction`; a query hitting a distractor does not; `reset()` twice in a row on the
      *same* instance with two different fixture rows shows zero state leakage (this directly
      exercises the pooled-instance-reuse behavior from CLAUDE.md's TRL mechanics section);
      `retrieval_fraction` caps at 1.0 even on a duplicate hit.
- [ ] `tests/unit/test_rewards.py` — fake `completions` (same message-list shape TRL actually
      uses) + duck-typed `environments` (objects exposing just `.retrieval_fraction`). Cover:
      well-formed correct answer + full retrieval; well-formed correct answer + zero retrieval;
      well-formed wrong answer; malformed/missing `<answer>` tag; hitting the hard tool-call cap
      mid-call (unresolved `tool_calls`, no answer). Assert exact reward values for both
      `get_reward_funcs("outcome_only")` and `get_reward_funcs("turn_level")`.
- [ ] Run `ruff check` / `ty check` (already configured as dev deps) and fix anything they flag.

## Exit criteria (all must be true before handing off)

- [ ] `pytest tests/unit/` passes, completes in well under a few seconds total, touches no
      network/GPU/live server.
- [ ] `ruff check` and `ty check` are clean.
- [ ] Every seam (retrieval HTTP call) is genuinely injectable — grep for any stray hardcoded
      `requests.post`/`httpx` call inside `env.py` that bypasses the injected client; there
      shouldn't be one.

## Handoff notes

<!-- Fill in after completing this phase: what was actually done, any deviations from the plan
above, gotchas hit (e.g. TRL API surface that didn't match CLAUDE.md's notes), and anything the
next phase's agent needs to know. Leave this section for the next fresh agent to read first. -->

(not yet started)
