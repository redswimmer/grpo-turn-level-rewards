# GRPO Turn Level Rewards

Experiments in turn-level reward shaping for GRPO training, built on
[`trl`](https://github.com/huggingface/trl) and tracked with
[`trackio`](https://github.com/gradio-app/trackio).

## Quick Start

Requires Python 3.13+, [`uv`](https://docs.astral.sh/uv/), and a JDK 21 (needed
by the retrieval server's Lucene bridge).

```bash
uv sync
```

### Retrieval server

Training and evaluation query a local BM25 retrieval server over the real
wiki-18 Wikipedia dump (see `CLAUDE.md`'s "Why this design" section for why).
Set it up once:

```bash
sudo apt install openjdk-21-jdk   # or your OS's JDK 21 equivalent; verify with `java -version`
uv sync
bash scripts/setup_retrieval.sh   # downloads the wiki-18 BM25 index (+corpus if needed) into data/wiki18/
```

`setup_retrieval.sh` prints the exact `retrieval_server.py` launch command to
use next (it depends on whether the downloaded index embeds raw documents —
for this project's confirmed download, it does not, so the corpus download is
needed too):

```bash
uv run python scripts/retrieval_server.py \
    --index_path data/wiki18/bm25-repo/bm25 \
    --corpus_path data/wiki18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
    --port 8000
```

Run it in the background or a separate terminal — it needs to stay up for the
rest of setup, and later for training/evaluation. Verify it's working:

```bash
uv run python scripts/verify_retrieval.py
```

This should print `PASS: retrieval server is up, wired correctly, and returns
real documents.` See `docs/phase-1-retrieval-infra.md`'s Handoff notes for
details on what was verified and a couple of corpus quirks worth knowing
about.

_No training entry point yet — usage instructions will land here once
`train.py` is added._

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
