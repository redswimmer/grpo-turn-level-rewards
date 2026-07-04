# grpo-turn-level-rewards

Experiments in turn-level reward shaping for GRPO training, built on
[`trl`](https://github.com/huggingface/trl) and tracked with
[`trackio`](https://github.com/gradio-app/trackio).

## Quick Start

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

_No runnable entry point yet — usage instructions will land here once training
scripts are added._

## Contributing

Install the git hooks (runs `ruff` and `ty` checks locally before each commit):

```bash
uv run pre-commit install
```

You can also run all checks manually at any time, against every file:

```bash
uv run pre-commit run --all-files
```

### Running quality gates directly

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
