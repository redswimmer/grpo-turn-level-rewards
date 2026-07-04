# Phase 2 Core Library — Design

Status: approved by user, 2026-07-04. Implements `docs/phase-2-core-library.md`.

## Context and evidence gathered

This design was grounded by reading actual code and data rather than assuming CLAUDE.md's prose
holds exactly:

- **Real dataset row** (`PeterJinGo/nq_hotpotqa_train`, streamed, first `hotpotqa` row) confirms
  the nested shape: `metadata = {"type", "level", "supporting_facts": {"title": [...], "sent_id":
  [...]}, "context": {"title": [...], "sentences": [...]}}`. `golden_answers` is a top-level list.
  Phase 3's own doc independently confirms this: eval-set rows must be reshaped to nest
  `supporting_facts`/`context` under `metadata` "matching the training set's nesting, since
  `env.py`/`rewards.py` should not need to know which source dataset a row came from." This
  settles `SearchEnv.reset()`'s contract — no need to guess or wait for Phase 3.
- **`scripts/retrieval_server.py`** (read directly) already runs `parse_title_text()` server-side
  before responding — the `/retrieve` response's `document` objects already have `title`/`text`
  fields split out. `env.py` must not re-derive them from `contents`; that logic already has a
  tested home in the server (it's exactly the bug Phase 1's handoff notes describe fixing there).
- **`transformers.utils.chat_template_utils.get_json_schema`** (read directly) raises
  `DocstringParsingException` if a tool method's docstring lacks an `Args:` description for any
  parameter — `SearchEnv.search()`'s docstring must satisfy this, not just be "documented".
- **`trl/trainer/grpo_trainer.py`**'s `_tool_call_loop` (read directly) confirms the exact
  completion message shapes: assistant tool-call messages are
  `{"role": "assistant", "content": ..., "tool_calls": [{"type": "function", "function":
  {"name": ..., "arguments": {...}}}]}`; tool responses are `{"role": "tool", "name": ...,
  "content": ...}`; hitting `max_tool_calling_iterations` leaves the last message as an
  unresolved assistant `tool_calls` message with no trailing tool/answer message. Test fixtures
  in `test_rewards.py` must match this exactly.
- **A live row's `prompt` column** turned out to be Search-R1's original text-tag ReAct prompt
  (`<search>...</search>` → `<information>...</information>` → `<answer>...</answer>`), which
  assumes a regex-based rollout loop, not TRL's native `environment_factory`/tool-calling
  (structured `tool_calls`, not text tags). **This is a gap the roadmap doesn't yet cover**:
  Phase 3 or 4 will need to replace this `prompt` column with one that teaches native tool use,
  keeping only the `<answer>...</answer>` convention for the final response. Noted here so it
  isn't rediscovered from scratch during Phase 4's smoke test; not a Phase 2 blocker since
  `rewards.py`/`env.py` don't depend on the system prompt's exact wording.

## Modules

### Package plumbing
- Add `[build-system]` (hatchling) to `pyproject.toml`; create `src/turn_level_rewards/__init__.py`.
- Add `pytest` to `[dependency-groups] dev`.
- `uv sync` twice (before and after adding pytest); confirm `uv run python -c "import
  turn_level_rewards"` succeeds and `uv.lock`'s source for this package is no longer `virtual`.

### `metrics.py`
Stdlib-only SQuAD-style: `normalize_answer` (lowercase, strip punctuation, drop articles
`a`/`an`/`the`, collapse whitespace), `exact_match`, `f1_score` (token-overlap F1; standard edge
case: if either normalized token list is empty, F1 is 1.0 only if both are empty, else 0.0).

### `env.py` — `SearchEnv`
```python
class SearchEnv:
    def __init__(self, retrieve_fn=None, topk=3, base_url="http://localhost:8000"): ...
    def reset(self, metadata, **kwargs) -> None:
        """Reset per-episode state. Pulls metadata['supporting_facts']['title']."""
    def search(self, query: str) -> str:
        """Search the wiki-18 corpus for relevant passages.

        Args:
            query (str): The search query.

        Returns:
            str: Formatted search results (title + text per hit).
        """
    @property
    def retrieval_fraction(self) -> float: ...
```
- `reset()`: dedups gold titles into `self._gold_titles: frozenset[str]`; resets `self._hit_titles
  = set()`. Called every episode via the pooled-instance reuse — must leave zero state from a
  prior episode.
- `search()`: calls `self._retrieve_fn(query, self._topk)` → `list[{"title", "text",
  "contents"}]`. Updates `self._hit_titles |= {titles in _gold_titles}`. Returns a numbered
  `Doc N (Title: "...")：text` string for the tool message; `"No results found."` if empty.
- Default `retrieve_fn` POSTs `{"queries": [query], "topk": topk, "return_scores": False}` to
  `f"{base_url}/retrieve"` and unwraps `result[0]`. This is the injectable seam — tests pass a
  plain function/dict, no HTTP.
- `retrieval_fraction`: `len(self._hit_titles) / len(self._gold_titles)` if gold titles exist,
  else `0.0`. No extra capping needed — hits are a subset of gold by construction.

### `rewards.py`
- `_extract_answer(completion) -> str | None`: regex over the last message's content for exactly
  one non-empty `<answer>...</answer>`; requires the last message to have no truthy `tool_calls`.
- `format_reward(completions, **kwargs)`: `+0.1` if `_extract_answer` succeeds, else `-0.1`.
- `outcome_reward(completions, golden_answers, **kwargs)`: `max` over `golden_answers` of
  `f1_score(extracted_or_empty, g) + (0.5 if exact_match(...) else 0)`.
- `turn_reward(environments, **kwargs)`: `0.4 * environment.retrieval_fraction`.
- `get_reward_funcs(condition: Literal["outcome_only", "turn_level"])`: `outcome_only` →
  `[format_reward, outcome_reward]`; `turn_level` → adds `turn_reward`. Raises `ValueError` on
  any other value (defends the CLI boundary; `Literal` alone doesn't stop a bad `--condition`
  string at runtime).

### Tests (`tests/unit/`)
- `test_metrics.py`: identical strings, partial overlap, case/punctuation/article normalization,
  fully disjoint answers.
- `test_env.py`: fake `retrieve_fn` (no HTTP). Cover: query hitting a gold title updates
  `retrieval_fraction`; query hitting a distractor does not; `reset()` twice in a row on the same
  instance with two different fixture rows shows zero state leakage; `retrieval_fraction` caps
  implicitly at 1.0 on a duplicate hit. Fixture titles: `"127 Hours"`, `"Big Stone Gap (film)"`,
  `"Peter Schmeichel"`, `"Virginia Commonwealth University"` (the four titles Phase 1's handoff
  verified live against the real corpus — not the CLAUDE.md examples that turned out not to
  exist).
- `test_rewards.py`: fake completions matching the exact TRL message shapes above (including the
  unresolved-`tool_calls`-at-cap case) + duck-typed environments (`.retrieval_fraction` only).
  Assert exact reward values for both `get_reward_funcs("outcome_only")` and
  `get_reward_funcs("turn_level")`.

### Validation loop (mirrors Phase 1's `verify_retrieval.py` pattern)
New `scripts/verify_phase2.py`: runs, in order, `uv run pytest tests/unit/`, `uv run ruff check`,
`uv run ty check`, and a check that the only `requests.post`/`httpx` call in `env.py` lives inside
the default `retrieve_fn` factory function (i.e. `SearchEnv.search()` itself never calls it
directly — it only ever calls `self._retrieve_fn`). Prints exactly which check failed (matching
`verify_retrieval.py`'s style: report the specific failing check, not just a generic failure), or
`PASS` and exits 0 if all pass.

Implementation loop: implement/adjust a module → run `scripts/verify_phase2.py` → on failure, fix
and re-run → repeat until `PASS` → only then update `docs/phase-2-core-library.md`'s Handoff notes
and the roadmap table in CLAUDE.md.

## Out of scope for this phase
- Replacing the dataset's `prompt` column with a native-tool-calling-oriented prompt (flagged
  above as a Phase 3/4 gap).
- `data.py` itself (Phase 3).
