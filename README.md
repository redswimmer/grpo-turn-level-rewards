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
comparison (`PPO`/`MT-PPO`).

## Results

The GRPO comparison (outcome-only vs. turn-level reward) has completed both training runs and a
full held-out evaluation (7,404 questions neither model trained on). Both conditions trained on
the identical number of steps and question-sampling process; only the reward function differs.

| Metric (held-out) | Outcome reward | Turn-level reward |
|---|---|---|
| Exact match | 0.236 | 0.207 |
| F1 | 0.331 | 0.294 |
| Well-formed answer rate | 0.994 | 0.974 |
| Real passage surfaced during search | n/a | 0.381 |

See `results/` for the full comparison plots (training curves, held-out bars, and the search-tool
call-frequency comparison referenced below).

**Honest read: this comparison doesn't currently support a claim in either direction.** Both
conditions show real learning (format compliance and EM/F1 both rose substantially from a
near-zero start, and outcome-only's held-out numbers match/exceed its own late-training numbers,
ruling out gross overfitting) — but three separate checks all point the same way:

- Turn-level reward looked ahead during training (EM/F1 both higher than outcome-only's), but that
  reverses on held-out data — outcome-only ends up ahead on both metrics instead.
- The held-out gap between the two conditions, either direction, is smaller than the noise already
  visible within a single condition's own training run.
- The paper's specific claimed mechanism — that outcome-only reward causes the model to gradually
  stop calling the search tool — didn't appear here. If anything, outcome-only's search-tool call
  frequency *rose* over training instead of falling.

At this scale (one training seed, 150 distinct training prompts per condition), that's the honest
outcome: a real, useful negative-ish result rather than a confirmation or refutation of the
paper's `GRPO-OR`/`GRPO-MR` finding. The recommended next step is a symmetric re-run — both
conditions get the same larger step budget and a new seed, never one condition alone — before
drawing a real conclusion; see `docs/phase-6-evaluation-comparison.md`'s Handoff notes for the
full criteria checked and the reasoning.

## Roadmap

- **GRPO: outcome-only vs. merged-reward** — training and held-out evaluation complete for both
  conditions; comparison inconclusive at this scale (see Results above) — a symmetric re-run at a
  larger step budget is recommended before drawing a conclusion.
- **PPO: outcome-only vs. merged-reward** — design complete; not yet started.
- **LLM-as-judge reward** (an alternative to exact-match/F1 scoring, explored on top of the PPO
  comparison) — not yet started.

## Project structure

```
data/       # downloaded wiki-18 retrieval corpus + BM25 index (gitignored, multi-GB)
docs/       # phase docs, design specs, roadmap
outputs/    # training checkpoints + logs per condition (gitignored)
results/    # final held-out metrics + comparison plots (committed)
scripts/    # retrieval server, one-off setup/verification, compare_runs.py
src/        # the turn_level_rewards package (env, rewards, metrics, data, train, evaluate)
tests/      # unit tests (fast, no GPU, no live retrieval server)
```

## Reproducing this

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
`--max-steps`, `--num-generations`, etc. explicitly for a full-scale run. Both conditions
log to the same [trackio](https://github.com/gradio-app/trackio) project
(`turn-level-rewards`) — run `trackio show --project turn-level-rewards` to view.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, quality gates, and running tests.
