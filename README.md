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

The GRPO comparison (outcome-only vs. turn-level reward) ran twice: an initial pair of training
runs (300 steps, 150 distinct training prompts each), then a symmetric re-run at double the
budget with a different seed (600 steps, 300 distinct prompts each) after the first pair came
back too noisy to trust. Both conditions in each run trained on the identical steps and
question-sampling process; only the reward function differs.

| Metric (held-out) | Outcome reward | Turn-level reward |
|---|---|---|
| Exact match | 0.242 | **0.307** |
| F1 | 0.343 | **0.399** |
| Well-formed answer rate | 0.986 | 0.892 |
| Real passage surfaced during search | n/a | 0.528 |

(Numbers above are the symmetric re-run, the trustworthy one — see `results/seed123_600steps/`
for its comparison plots. The original run's numbers are in `results/`.)

**The symmetric re-run resolves what the first run couldn't: turn-level reward shows a real,
held-out-confirmed advantage over outcome-only reward, in the direction the source paper reports.**
The first run was inconclusive — turn-level reward looked ahead during training, but that reversed
on held-out data, and the gap either way was smaller than single-run noise. Doubling the training
data and using a different seed fixed that: turn-level reward now leads during training *and* on
held-out data (+0.065 EM, +0.056 F1), and the concerning decline in "real passage surfaced during
search" from the first run reversed too (it now *rises* over training, 0.40→0.57). This isn't a
reproduction of the paper's exact numbers (this repro uses a much smaller model, a different
outcome-reward formula, and a fraction of their likely training scale — see
`docs/phase-6-evaluation-comparison.md` for the full list of documented deviations), but it's a
real, evidence-based positive signal for their core claim, not an overclaim off noisy data.

One thing that *didn't* resolve: outcome-only reward's search-tool call frequency still rises
over training instead of falling, the opposite of the paper's claimed mechanism for why
outcome-only reward underperforms. Two follow-up experiments are testing specific hypotheses
about the training setup — a length penalty (completions grew ~4x with no accuracy benefit, a
free-riding side effect nothing in the reward currently discourages) and a search-count penalty
(replacing the prompt-engineered "at most 2 searches" instruction with a reward-shaped
constraint, borrowing a mechanism from the paper's separate PPO design — not a GRPO paper
reproduction, since their GRPO ablation doesn't use this mechanism at all). Results pending.

## Roadmap

- **GRPO: outcome-only vs. merged-reward** — training and held-out evaluation complete for both
  conditions across two runs; the symmetric re-run shows a real, held-out-confirmed advantage for
  turn-level reward (see Results above). Two follow-up reward-design experiments in progress.
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
