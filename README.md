# turn-level-rewards

## Development setup

Install dependencies:

```bash
uv sync
```

Install the git hooks (runs ruff and ty checks locally before each commit):

```bash
uv run pre-commit install
```

You can also run all checks manually at any time, against every file:

```bash
uv run pre-commit run --all-files
```
