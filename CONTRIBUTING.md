# Contributing

## Setup

```bash
uv sync --all-groups
```

Install the git hooks (runs `ruff` and `ty` checks locally before each commit):

```bash
uv run pre-commit install
```

You can also run all checks manually at any time, against every file:

```bash
uv run pre-commit run --all-files
```

## Running quality gates directly

For faster iteration, you can run the individual tools without going through
`pre-commit`:

```bash
# Lint (add --fix to auto-fix what's fixable)
uv run ruff check .

# Format (drop --check to apply formatting instead of just checking it)
uv run ruff format --check .

# Type check
uv run ty check
```

## Running tests

```bash
uv run pytest tests/unit/
```

Tests in `tests/unit/` are fast and deterministic — no GPU, live retrieval server, or network
access required (external boundaries like the retrieval HTTP call are injected and faked in
tests). See `CLAUDE.md`'s "Guiding principles for code, tests, and dependencies" section for why
the repo is scoped to a single test tier.
