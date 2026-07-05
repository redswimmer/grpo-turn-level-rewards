# Outcome vs. Turn-Level Reward for Multi-Turn Search Agents

A small-scale experiment testing whether rewarding an AI agent's intermediate actions — not just
its final answer — helps it learn faster and more reliably.

## What this compares

This repo trains a multi-turn Wikipedia-search agent under two reward regimes:

- **Outcome reward** — the agent is scored only on its final answer's correctness. Sparse: no
  signal until the very end of the episode.
- **Turn-level reward** — the same outcome scoring, plus a bonus for surfacing a real
  supporting-fact passage during search. Denser: the agent gets credit for good intermediate
  behavior, not just a good final answer.

The interesting question isn't just "does the denser signal help" — it's **whether that holds up
across genuinely different reinforcement learning algorithms**, not just one. So this repo tests
the same outcome-vs-turn-level comparison twice:

- **GRPO** — scores a group of the agent's attempts at the same question against each other,
  using the ones that did relatively better within the group as the learning signal.
- **PPO** — learns a running estimate of how good a position is (a value function), and nudges
  the policy toward actions that beat that estimate, turn by turn.

If turn-level reward helps in the same way under both, that's a real finding about reward shaping
in multi-turn agent RL, not an artifact of one algorithm's mechanics.

Concretely, this is a simplified reproduction of two ablations from ["Reinforcing Multi-Turn
Reasoning in LLM Agents via Turn-Level Reward Design"](https://arxiv.org/abs/2505.11821)
(arXiv:2505.11821): its Appendix E GRPO case study (`GRPO-OR`/`GRPO-MR`), and its main-results PPO
comparison (`PPO`/`MT-PPO`). See `CLAUDE.md`'s Roadmap for what's built and what's still in
progress. (`MT-GRPO`, a further turn-level credit-assignment scheme specific to GRPO, remains out
of scope — see `CLAUDE.md` for why.)

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

### Training

```bash
uv run python -m turn_level_rewards.train --condition outcome_only
uv run python -m turn_level_rewards.train --condition turn_level
```

The bare invocation above (no extra flags) runs at smoke-test scale — 8 rows, 2 steps, a real
`Qwen/Qwen3.5-0.8B` model against the retrieval server started above. Pass `--train-size`,
`--max-steps`, `--num-generations`, etc. explicitly for a full-scale run; see
`docs/phase-5-full-training-runs.md` for a paper-grounded example configuration. Both conditions
log to the same [trackio](https://github.com/gradio-app/trackio) project
(`turn-level-rewards`) — run `trackio show --project turn-level-rewards` to view.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, quality gates, and running tests.
