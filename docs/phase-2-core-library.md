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

- **Phase 1 is complete.** The retrieval server is verified working — see
  `docs/phase-1-retrieval-infra.md`'s Handoff notes for the exact launch command (needed later
  for the live smoke test in Phase 4, not this phase), the `contains_doc=False` result, and a
  `retrieval_server.py` bug that was found and fixed there.
- This phase only needs the documented request/response **contract** (JSON shape), not a live
  running server — tests inject a fake retriever at that seam (see CLAUDE.md's "Guiding
  principles", point 1). **The contract itself did not change** during Phase 1 — the bug fixed
  there made the corpus-fallback path match the contract already documented in CLAUDE.md, not a
  new one — but skim those Handoff notes once anyway before assuming the CLAUDE.md contract is
  exactly right, since that's where any such change would have been recorded.
- Also worth knowing from Phase 1 (not a contract change, but relevant if this phase's tests use
  realistic fixture data): the wiki-18 corpus has real gaps and near-duplicate titles — e.g. the
  `retrieval_fraction` accounting in `env.py`'s tests should use fixture titles/documents that
  look like genuine wiki-18 rows (`{title, text, contents}` with `contents = '"<Title>"\n<text>'`
  per CLAUDE.md's confirmed schema), not assume every gold title is guaranteed retrievable.

## Tasks

- [x] **Package plumbing (do this first, before writing any reward/env logic)**: this repo is
      currently a uv *virtual* project — `pyproject.toml` has no `[build-system]` table, and
      `uv.lock` has `source = { virtual = "." }` — so there is no installed `turn_level_rewards`
      package yet and `src/` doesn't exist. Before `import turn_level_rewards` (from tests, or
      later from `train.py`) can work, you need to make this project installable:
      - Add a `[build-system]` table (`hatchling` is the natural choice — it auto-detects a
        `src/<project-name-with-underscores>/` layout with no extra config needed beyond that).
      - Create `src/turn_level_rewards/__init__.py`.
      - Run `uv sync` again and confirm `uv run python -c "import turn_level_rewards"` succeeds,
        and that `uv.lock`'s `source` for this package is no longer `virtual`.
      - Add `pytest` to `[dependency-groups] dev` in `pyproject.toml` (it is not there yet —
        confirmed by grepping `pyproject.toml`/`uv.lock`, don't assume it's already available)
        and run `uv sync` once more.
- [x] `src/turn_level_rewards/metrics.py`: `normalize_answer`, `exact_match`, `f1_score`
      (SQuAD-style, stdlib only — no new dependencies for this file).
- [x] `src/turn_level_rewards/env.py`: `SearchEnv` class.
      - `reset(self, metadata, **kwargs) -> None` — resets all mutable per-episode state
        (remember: TRL reuses instances from a pool; leftover state from a prior episode is a
        real bug class here, not a hypothetical). **As actually built** (see Handoff notes): this
        takes the row's nested `metadata` dict directly (`metadata["supporting_facts"]["title"]`),
        not flat `context`/`supporting_facts` kwargs as originally sketched here — confirmed
        against a real dataset row before implementation, not assumed.
      - `search(self, query: str) -> str` tool method — calls the retrieval server. **The HTTP
        client/call itself must be injectable** (constructor parameter, module-level default
        factory, or similar) — this is the seam principle 1 in CLAUDE.md refers to concretely.
        **As actually built**: trusts `document["title"]`/`document["text"]` directly from the
        retrieval server's response — no `contents.split(...)` re-parsing in `env.py`, since the
        server already does that parsing itself (confirmed by reading `retrieval_server.py`).
      - Tracks `retrieval_fraction` (dedup'd fraction of gold `supporting_facts` titles actually
        surfaced), capped at 1.0.
- [x] `src/turn_level_rewards/rewards.py`:
      - `format_reward(completions, **kwargs)`
      - `outcome_reward(completions, golden_answers, **kwargs)` — SQuAD F1 + EM bonus, maxed over
        the `golden_answers` list (it's a list, not a single string — see CLAUDE.md's Dataset
        section).
      - `turn_reward(environments, **kwargs)` — `0.4 * environment.retrieval_fraction`.
      - `get_reward_funcs(condition: Literal["outcome_only", "turn_level"])` — returns the right
        list per CLAUDE.md's Reward design section.
      - All functions operate on plain data (strings/dicts/duck-typed objects) — no real
        `GRPOTrainer` or loaded model needed to exercise them.
- [x] `tests/unit/test_metrics.py` — EM/F1 sanity pairs (identical strings, partial overlap,
      case/punctuation/article normalization, fully disjoint answers).
- [x] `tests/unit/test_env.py` — inject a **fake retriever** (plain function/dict returning
      canned documents, no real HTTP). Cover: a query that should hit a gold title updates
      `retrieval_fraction`; a query hitting a distractor does not; `reset()` twice in a row on the
      *same* instance with two different fixture rows shows zero state leakage (this directly
      exercises the pooled-instance-reuse behavior from CLAUDE.md's TRL mechanics section);
      `retrieval_fraction` caps at 1.0 even on a duplicate hit.
- [x] `tests/unit/test_rewards.py` — fake `completions` (same message-list shape TRL actually
      uses) + duck-typed `environments` (objects exposing just `.retrieval_fraction`). Cover:
      well-formed correct answer + full retrieval; well-formed correct answer + zero retrieval;
      well-formed wrong answer; malformed/missing `<answer>` tag; hitting the hard tool-call cap
      mid-call (unresolved `tool_calls`, no answer). Assert exact reward values for both
      `get_reward_funcs("outcome_only")` and `get_reward_funcs("turn_level")`.
- [x] Run `ruff check` / `ty check` (already configured as dev deps) and fix anything they flag.

## Exit criteria (all must be true before handing off)

- [x] `uv run pytest tests/unit/` passes, completes in well under a few seconds total, touches no
      network/GPU/live server.
- [x] `ruff check` and `ty check` are clean.
- [x] Every seam (retrieval HTTP call) is genuinely injectable — grep for any stray hardcoded
      `requests.post`/`httpx` call inside `env.py` that bypasses the injected client; there
      shouldn't be one.

## Handoff notes

- **`scripts/verify_phase2.py`** is the exit-criteria gate (mirrors `verify_retrieval.py`'s
  pattern) — re-run it after any future change to `metrics.py`/`env.py`/`rewards.py`. Confirmed
  `PASS` as of this handoff.
- **`SearchEnv.reset()`'s exact contract**: `reset(self, metadata, **kwargs)`, pulling
  `metadata["supporting_facts"]["title"]`. Confirmed directly against a real streamed row from
  `PeterJinGo/nq_hotpotqa_train` (a nested `metadata` dict, not a flat `context`/`supporting_facts`
  kwargs shape) and independently corroborated by Phase 3's own doc, which already commits to
  nesting `supporting_facts`/`context` under `metadata` for the eval set too. Phase 3's `data.py`
  needs no special-casing here — just pass `metadata` through as-is.
- **The retrieval server already parses `title`/`text` server-side**
  (`parse_title_text()` in `scripts/retrieval_server.py`, confirmed by reading it directly) —
  `SearchEnv.search()` trusts `document["title"]`/`document["text"]` directly and does no
  re-parsing of `contents`.
- **Answer format**: final answers are wrapped in `<answer>...</answer>` in the last assistant
  message (no unresolved `tool_calls`), checked by `rewards._extract_answer`. This convention is
  reused, unmodified, from this dataset's own baked-in `prompt` column.
- **Known gap for Phase 3/4, not fixed here — now tracked explicitly as a task in those phase
  docs**: `PeterJinGo/nq_hotpotqa_train`'s `prompt` column (confirmed by pulling a real row) is
  Search-R1's original text-tag ReAct prompt (`<search>...</search>` →
  `<information>...</information>` → `<answer>...</answer>`), which assumes a regex-based rollout
  loop — not TRL's native `environment_factory` tool-calling (structured `tool_calls`, not text
  tags). Phase 3/4 replaces this `prompt` column with one that teaches native tool use, keeping
  only the `<answer>` convention for the final response. `rewards.py`/`env.py` do not depend on
  the exact prompt wording, so this did not block Phase 2.
- **Terminology note (added after Phase 2, during scope-reframing discussion)**: this repo's
  `outcome_only`/`turn_level` conditions map to the paper's `GRPO-OR`/`GRPO-MR` respectively — see
  `CLAUDE.md`'s Goal section for the full mapping and what's explicitly out of scope (`MT-GRPO`,
  `PPO`/`MT-PPO`).
- **Fixture titles**: `test_env.py` uses `"127 Hours"`, `"Big Stone Gap (film)"`,
  `"Peter Schmeichel"`, `"Virginia Commonwealth University"` — the four titles Phase 1's handoff
  notes verified live against the real corpus, not the CLAUDE.md examples that turned out not to
  exist.
- **Merged to `main` via PR #2.**
