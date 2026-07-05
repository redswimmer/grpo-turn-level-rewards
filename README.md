# Outcome-Only vs. Merged-Reward GRPO

A small-scale experiment testing whether rewarding an AI agent's intermediate actions — not just
its final answer — helps it learn faster and more reliably.

## What this compares

This repo tests that question with GRPO, on a multi-turn Wikipedia-search agent, by training two
otherwise-identical models that differ only in reward shaping.

Concretely, it's a simplified reproduction of one ablation from ["Reinforcing Multi-Turn
Reasoning in LLM Agents via Turn-Level Reward Design"](https://arxiv.org/abs/2505.11821)
(arXiv:2505.11821) — specifically its Appendix E case study, stated in the paper's own terms:

- **`GRPO-OR`** (Outcome Reward): reward = final-answer correctness + format only.
- **`GRPO-MR`** (Merged Reward): the same, plus a bonus for surfacing a real supporting-fact
  passage during search.

Both conditions run the exact same multi-turn Wikipedia-search agent and the exact same
underlying RL algorithm — only the reward signal changes between the two training runs. The
paper describes two further variants, `MT-GRPO` and `PPO`/`MT-PPO`, that go further by changing
the *algorithm* itself, not just the reward — those are out of scope for this pass. See
`CLAUDE.md` for what they are and why they're not attempted here.

## Quick Start

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- JDK 21 (needed by the retrieval server's Lucene bridge)

```bash
uv sync
sudo apt install openjdk-21-jdk
```

### Retrieval server

Training and evaluation search a local BM25 server backed by the real wiki-18
Wikipedia dump (~21M passages). Set it up once:

```bash
bash scripts/setup_retrieval.sh   # downloads the wiki-18 BM25 index (+corpus if needed) into data/wiki18/
```

The script downloads the index, checks whether it also needs the separate
corpus file, and prints the exact command to launch the server — something
like:

```bash
uv run python scripts/retrieval_server.py \
    --index_path data/wiki18/bm25-repo/bm25 \
    --corpus_path data/wiki18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
    --port 8000
```

Run that (in the background or a separate terminal — it needs to stay up for
the rest of setup and for training/evaluation later), then confirm it's
working:

```bash
uv run python scripts/verify_retrieval.py
```

```
PASS: retrieval server is up, wired correctly, and returns real documents.
```

_No training entry point yet — usage instructions will land here once
`train.py` is added._

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, quality gates, and running tests.
