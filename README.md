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

**Status: the GRPO comparison below (outcome reward vs. turn-level reward, Results 1–3) is
complete.** The PPO comparison described above has a finished design but hasn't been run yet —
see Roadmap. Everything below is GRPO-only.

### 1. What's actually being measured

The agent answers multi-hop trivia questions (from HotpotQA's validation split — 7,404 questions
neither reward condition ever trained on) by searching a real ~21M-passage Wikipedia snapshot and
producing a final answer. Three metrics track different things:

- **Exact match (EM)** — did the agent's final answer literally match an accepted answer string?
  Strict: "Barack Obama" ≠ "Obama."
- **F1** — word-overlap partial credit (the standard SQuAD-style scoring paper QA benchmarks use)
  for answers that are close but not verbatim.
- **Retrieval fraction** — of the real supporting-fact passages actually needed to answer the
  question, what fraction did the agent's searches surface? Only meaningful for turn-level reward,
  since that's the only condition whose reward depends on it.

### 2. Turn-level reward wins — consistent with the source paper's direction, smaller in magnitude

![Held-out exact match and F1: outcome reward vs. turn-level reward](results/held_out_em_f1_comparison.png)

| Metric (held-out) | Outcome reward | Turn-level reward | Source paper's `GRPO-OR` / `GRPO-MR` (Table 2) |
|---|---|---|---|
| Exact match | 0.242 | **0.307** | 0.0 / **0.335** |
| F1 | 0.343 | **0.399** | not reported |
| Retrieval fraction | n/a | 0.528 | not reported |

**Consistent with the paper**: turn-level reward beats outcome reward in both — the direction the
paper reports holds up here too. **Deviates in two ways worth being explicit about:**

- *Magnitude.* Our numbers are both lower and closer together than the paper's. This repo uses a
  far smaller model (0.8B vs. the scale papers like this typically train at) and a fraction of the
  likely training data — a much lower ceiling on absolute performance is expected, not a red flag.
- *Outcome reward's total failure didn't reproduce.* The paper's `GRPO-OR` scores a stark **0.0**
  exact match — total collapse. Ours scores a real, non-trivial 0.242. The most likely reason,
  documented and deliberate: the paper's outcome reward is pure binary exact-match (1.0 or 0.0,
  nothing in between), while this repo's adds F1 partial credit specifically because a group of
  rollouts that all score a flat 0 gives GRPO's relative-ranking signal nothing to learn from —
  partial credit avoids that trap. So our outcome-only agent actually learns something, where the
  paper's apparently didn't. This is a genuine, acknowledged difference in setup, not a
  reproduction of their exact ablation.

**One further mechanism claim from the paper did *not* reproduce**: the paper states `GRPO-OR`
"gradually stops calling search tools" over training. Here, outcome reward's search-call frequency
*rises* over training instead of declining — the opposite trend. This is a real, unresolved
discrepancy (not explained away), independent of the main EM/F1 result above.

<details>
<summary>Is the EM/F1 win just favorable timing, or does it hold up throughout training?</summary>

![Smoothed training curves: exact match and F1 over training steps](results/training_curves_smoothed.png)

Turn-level reward (orange) leads for most of training, not just at the final checkpoint — this
rules out "got lucky at the end" as the explanation. (Curves are a 15-point rolling average of
per-step training metrics; the raw values are noisy step-to-step, as GRPO reward inherently is —
smoothing is only for readability, not a different underlying result.)

One methodological note for the curious: this result needed two attempts. A first, smaller run
(300 steps) came back too noisy to trust — turn-level reward looked ahead during training but that
reversed on held-out data. Doubling the training budget and re-running with a different seed
resolved it, with turn-level reward leading on both training *and* held-out data. Full numbers for
both runs are in `docs/phase-6-evaluation-comparison.md`.
</details>

### 3. Naive attempts to improve it further backfired — and that's the more interesting finding

The natural next question: can we push turn-level reward's advantage further, or fix outcome
reward's remaining weaknesses, with a bit more reward engineering? Three experiments tried. **None
worked** — but the *way* they failed is the actual lesson here.

![Held-out exact match across all four reward configurations](results/followup_experiments_comparison.png)

- **A length penalty** (discourage long completions) **collapsed outcome reward completely** —
  the model stopped searching and started producing incoherent, garbled text. Turn-level reward
  survived, with only a modest real cost.
- **A search-count penalty** (punish each search call directly, borrowed from the source paper's
  separate PPO design) was **even worse for outcome reward** — same total collapse, but this time
  the final answers were nonsense strings, not just wrong. Turn-level reward also took damage this
  time (it collapsed too, for about 70% of training) but *recovered* in the final stretch — outcome
  reward never did.
- **A control experiment** — removing the original prompt instruction ("search at most twice")
  with *no* reward penalty at all — isolated why: outcome reward searched *more*, not less,
  without that instruction, and paid only a small accuracy cost. So the two penalty experiments'
  collapses weren't about losing guidance — they were caused by the penalty itself.

**Why this happens, in plain terms:** GRPO scores a batch of the model's attempts at one question
purely *relative to each other* — there's no separate "how good is this really" estimate to fall
back on, the way PPO's value function provides. So if every attempt in a batch stumbles onto the
same cheap trick — "just guess something, don't bother searching, accept the penalty is smaller
than the risk of a real answer" — GRPO has no way to see past that shared blind spot. It doesn't
just fail to punish the trick; it can't even tell the trick happened, because everything in the
batch looks equally (un)rewarding. Outcome reward's simple two-part scoring (nail the format, get
the answer right) gave the model nowhere else to go once a penalty made honest effort look risky.
Turn-level reward's extra signal — credit for good search behavior specifically — gave the model
something to hold onto even under penalty pressure, which is why it bent instead of breaking.

**The takeaway for anyone shaping a reward function under GRPO specifically**: adding a bare
penalty term without a matching positive incentive pulling toward the behavior you actually want
is genuinely risky — more risky than the same change might be under an algorithm with a value
function (like PPO) to catch a whole batch making the same mistake. A denser, more structured
reward isn't just "more accurate" — it's also more robust to your own future changes.

Full numbers, example completions from the collapses, and the complete methodology are in
`docs/phase-6-evaluation-comparison.md` — not required reading, everything above is the full
story.

## Roadmap

- **GRPO: outcome-only vs. merged-reward** — training and held-out evaluation complete for both
  conditions across two runs; the symmetric re-run shows a real, held-out-confirmed advantage for
  turn-level reward (see Results above). Three follow-up reward-design experiments are complete
  (see Results above); Phase 6 is fully done.
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
