"""train_ppo.py: custom multi-turn PPO trainer (MTPPOTrainer) for ppo/mt_ppo conditions.

Built directly on transformers.Trainer, not GRPOTrainer/PPOTrainer -- TRL's PPOTrainer has no
multi-turn tool-calling support (confirmed fresh against the installed 1.7.1 and upstream's
dev branch, re-verified 2026-07-23; see
docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md). Reuses SearchEnv/rewards.py/data.py
unmodified. See CLAUDE.md's Goal section and docs/phase-7-mt-ppo.md for the full design.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response

from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import TURN_REWARD_SCALE

Condition = Literal["ppo", "mt_ppo"]

MODEL_NAME = "Qwen/Qwen3.5-0.8B"


@dataclass
class MTPPOConfig(TrainingArguments):
    """Config for MTPPOTrainer. Subclasses TrainingArguments the same way TRL's GRPOConfig does,
    adding this trainer's own fixed hyperparameters (see this plan's Global Constraints section
    for where each value comes from).
    """

    condition: Condition = "ppo"
    n_max: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.001
    policy_lr: float = 1e-6
    critic_lr: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    num_ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    num_rollouts_per_step: int = 2
    max_completion_length: int = 2048
    project: str = "turn-level-rewards-ppo"


def build_ppo_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_rollouts_per_step: int,
) -> MTPPOConfig:
    """Build the MTPPOConfig for a training run. Mirrors train.py's build_config role from
    Phase 4 -- fixed hyperparameters are baked in here, not exposed as independent CLI flags.
    """
    return MTPPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        condition=condition,
        n_max=4,
        clip_eps=0.2,
        kl_beta=0.001,
        policy_lr=1e-6,
        critic_lr=1e-5,
        gamma=1.0,
        gae_lambda=1.0,
        num_ppo_epochs=4,
        value_loss_coef=0.5,
        num_rollouts_per_step=num_rollouts_per_step,
        max_completion_length=2048,
        logging_steps=1,
        run_name=condition,
        report_to="none",  # trackio is called directly in MTPPOTrainer.train(), not through
        # transformers' generic report_to integration -- see Task 8.
    )


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 1.0,
    lam: float = 1.0,
    bootstrap_value: float = 0.0,
) -> list[float]:
    """Generalized Advantage Estimation (standard recursive formula).

    len(values) must equal len(rewards) -- values[t] is the critic's estimate at position t.
    bootstrap_value is V for the (terminal) state after the last reward -- 0.0 for an episode
    that truly ends, since there's no further return to bootstrap from. At this repo's fixed
    gamma=1, lambda=1 (paper's own spec), this reduces toward a full-episode
    Monte-Carlo-return-minus-baseline -- no discount/decay tuning needed.
    """
    if len(rewards) != len(values):
        raise ValueError(
            f"rewards ({len(rewards)}) and values ({len(values)}) must be equal length"
        )
    advantages = [0.0] * len(rewards)
    running_gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running_gae = delta + gamma * lam * running_gae
        advantages[t] = running_gae
        next_value = values[t]
    return advantages


def place_turn_rewards(
    num_tokens: int,
    turn_boundary_token_indices: list[int],
    retrieval_fraction_after_each_turn: list[float],
    format_and_outcome_reward: float,
    condition: Condition,
    turn_reward_scale: float = TURN_REWARD_SCALE,
) -> list[float]:
    """Eq. 9 turn-boundary reward placement.

    R^O (format_reward + outcome_reward, summed by the caller) always lands on the trajectory's
    last token. R^I -- turn_reward's marginal per-turn contribution -- lands at each intermediate
    turn boundary, mt_ppo only; always 0 for ppo (single lump-sum credit assignment even across a
    multi-turn episode, per the paper's Eq. 9).

    turn_boundary_token_indices and retrieval_fraction_after_each_turn operate over whatever
    token-index space the caller is using (this repo's MTPPOTrainer uses action-token-relative
    indices, i.e. only counting policy-generated tokens -- see _rollout_episode's docstring).
    retrieval_fraction_after_each_turn[i] is SearchEnv.retrieval_fraction sampled immediately
    after intermediate turn i's tool call executed. retrieval_fraction is monotonically
    non-decreasing (SearchEnv only ever adds to its hit set), so each turn's real, marginal
    contribution is that turn's value minus the previous turn's (0.0 before the first turn) --
    not the raw cumulative value, which would double-count every later turn's reward.
    """
    if len(turn_boundary_token_indices) != len(retrieval_fraction_after_each_turn):
        raise ValueError(
            "turn_boundary_token_indices and retrieval_fraction_after_each_turn must be equal "
            "length"
        )
    per_token_rewards = [0.0] * num_tokens
    per_token_rewards[-1] += format_and_outcome_reward
    if condition == "mt_ppo":
        previous_fraction = 0.0
        for token_index, cumulative_fraction in zip(
            turn_boundary_token_indices, retrieval_fraction_after_each_turn, strict=True
        ):
            marginal = cumulative_fraction - previous_fraction
            per_token_rewards[token_index] += turn_reward_scale * marginal
            previous_fraction = cumulative_fraction
    return per_token_rewards


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.001,
    value_loss_coef: float = 0.5,
) -> dict[str, torch.Tensor]:
    """PPO-clip policy loss + value loss + a direct KL penalty term, all masked to action tokens.

    action_mask is 1.0 at positions that are real policy-generated action tokens, 0.0 elsewhere
    (prompt tokens, tool-response tokens injected by the environment, padding) -- none of those
    should ever receive policy gradient. The KL term uses old_logprobs (the rollout-time, frozen
    policy snapshot) as the reference throughout every one of this batch's inner PPO epochs --
    not a separate frozen reference model the way GRPO's beta works (see the design spec's
    stated assumption on this point). Both the clip and the KL term are applied together, not
    either/or, per the paper's spec.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -_masked_mean(torch.min(unclipped, clipped), action_mask)
    value_loss = _masked_mean((new_values - returns) ** 2, action_mask)
    kl = _masked_mean(new_logprobs - old_logprobs, action_mask)
    loss = policy_loss + value_loss_coef * value_loss + kl_beta * kl
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.detach(),
    }


class _PolicyAndCritic(nn.Module):
    """Wraps the policy (causal LM) and critic (sequence-classification head) as one nn.Module.

    Lets Trainer's standard model/optimizer/checkpoint plumbing see both submodules through a
    single self.model, while create_optimizer still gives them independent learning rates (the
    paper's own spec: policy_lr=1e-6, critic_lr=1e-5) via separate param groups.
    """

    def __init__(self, policy: PreTrainedModel, critic: PreTrainedModel) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic


def build_policy_and_critic(model_name: str = MODEL_NAME) -> _PolicyAndCritic:
    """Real policy + real critic, both loaded from the same base checkpoint -- separate models,
    not a shared backbone (the paper's own spec). Not unit-tested: loads real weights: validated
    by the live smoke test (Task 11) instead.
    """
    policy = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    critic = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, dtype=torch.bfloat16
    )
    return _PolicyAndCritic(policy, critic)


class MTPPOTrainer(Trainer):
    """Custom multi-turn PPO trainer with tool-calling, built directly on transformers.Trainer.

    Owns: the rollout loop (render with tools -> generate -> parse_response -> execute
    SearchEnv.search() on a tool call -> append tool message -> repeat up to args.n_max turns ->
    require a final <answer>), the critic forward pass, Eq. 9 reward placement, GAE, and the
    PPO-clip + KL-penalty + value-loss update. Turn-level credit assignment for mt_ppo falls out
    of reward placement + GAE bootstrapping alone -- no MT-GRPO-style extra-rollout advantage
    trick is needed here (PPO already has a real per-token critic). See
    docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md's Context section for why this is
    built on transformers.Trainer directly rather than subclassing GRPOTrainer/PPOTrainer.
    """

    def __init__(
        self,
        condition: Condition,
        model: _PolicyAndCritic,
        tokenizer,
        train_dataset,
        args: MTPPOConfig,
        environment_factory=None,
        callbacks=None,
    ) -> None:
        self.condition = condition
        self.environment_factory = environment_factory or SearchEnv
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=callbacks)
        self.tokenizer = add_response_schema(tokenizer)
        self.training_chat_template = get_training_chat_template(self.tokenizer)

    # create_optimizer(self) intentionally omits the base transformers.Trainer.create_optimizer's
    # optional `model` parameter. The internal Trainer code path that passes it positionally
    # is only reached for FSDP-XLA, SageMaker MP-DP, or DataParallel; this repo's single-GPU
    # setup (per CLAUDE.md's Hardware section) never triggers it, so the omission is safe.
    def create_optimizer(self) -> torch.optim.Optimizer:  # ty: ignore[invalid-method-override]
        """One AdamW, two param groups -- policy_lr / critic_lr per the paper's spec (10x apart)."""
        # self.model.policy, self.model.critic, self.args.policy_lr, and self.args.critic_lr
        # are all real attributes defined on this file's _PolicyAndCritic and MTPPOConfig
        # classes respectively. They are safe; ty's inability to see through Trainer's looser
        # base types is not a real issue here.
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.policy.parameters(), "lr": self.args.policy_lr},  # ty: ignore[unresolved-attribute]
                {"params": self.model.critic.parameters(), "lr": self.args.critic_lr},  # ty: ignore[unresolved-attribute]
            ]
        )
        return self.optimizer

    def _rollout_episode(self, row: dict) -> dict:
        """Run one multi-turn episode for a single dataset row: real generation, real tool calls
        against the real retrieval server (via self.environment_factory, e.g. SearchEnv).

        Returns everything downstream needs, all expressed over a "compressed action-token"
        index space that only counts policy-generated (assistant) tokens -- prompt tokens and
        tool-response tokens (environment-injected, not sampled by the policy) are excluded from
        this space entirely, matching how GAE/PPO treat each policy-generated token as one RL
        timestep:
          - full_token_ids: every token in the final rendered conversation (prompt + all turns),
            in order -- fed to the policy/critic for full context.
          - action_mask: same length as full_token_ids, 1 at positions the policy generated,
            0 elsewhere.
          - turn_boundary_action_indices: for each intermediate turn (one that made a tool call,
            i.e. not the final answering turn), the index INTO THE COMPRESSED ACTION-TOKEN
            SEQUENCE (not full_token_ids) of that turn's last generated token.
          - retrieval_fraction_after_each_turn: SearchEnv.retrieval_fraction sampled immediately
            after that same turn's tool call executed -- one entry per turn_boundary_action_index,
            same order.

        Relies on get_training_chat_template's prefix-preserving guarantee (confirmed supported
        for Qwen3.5): each turn's freshly-rendered prompt is guaranteed to start with exactly the
        tokens already recorded in full_token_ids, so the new suffix at each turn is unambiguous.
        """
        environment = self.environment_factory()
        environment.reset(**row)
        messages = list(row["prompt"])
        # self.model is this file's _PolicyAndCritic and self.args is this file's MTPPOConfig at
        # runtime (both set in __init__/composed by build_ppo_config), but Trainer's base type
        # stubs only know them as the looser `nn.Module | None` / `TrainingArguments` -- same
        # ty-can't-see-through-Trainer's-base-types situation already noted on create_optimizer
        # above. Safe to ignore here for the same reason.
        policy = self.model.policy  # ty: ignore[unresolved-attribute]

        full_token_ids: list[int] = []
        action_mask: list[int] = []
        turn_boundary_action_indices: list[int] = []
        retrieval_fraction_after_each_turn: list[float] = []
        num_action_tokens = 0

        for _turn in range(self.args.n_max):  # ty: ignore[unresolved-attribute]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tools=[environment.search],
                add_generation_prompt=True,
                chat_template=self.training_chat_template,
                tokenize=False,
            )
            prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            new_context_tokens = prompt_token_ids[len(full_token_ids) :]
            full_token_ids.extend(new_context_tokens)
            action_mask.extend([0] * len(new_context_tokens))

            input_ids = torch.tensor([prompt_token_ids], device=policy.device)  # ty: ignore[invalid-argument-type]
            with torch.no_grad():
                generation = policy.generate(  # ty: ignore[call-non-callable, unresolved-attribute]
                    input_ids,
                    max_new_tokens=self.args.max_completion_length,  # ty: ignore[unresolved-attribute]
                    do_sample=True,
                    temperature=1.0,
                )
            new_token_ids = generation[0, len(prompt_token_ids) :].tolist()
            parsed = parse_response(self.tokenizer, new_token_ids, prefix=prompt_token_ids)
            messages.append(parsed)

            full_token_ids.extend(new_token_ids)
            action_mask.extend([1] * len(new_token_ids))
            num_action_tokens += len(new_token_ids)

            tool_calls = parsed.get("tool_calls") or []
            if not tool_calls:
                break  # final answer turn -- episode complete

            for tool_call in tool_calls:
                result = environment.search(**tool_call["function"]["arguments"])
                messages.append({"role": "tool", "name": "search", "content": result})

            turn_boundary_action_indices.append(num_action_tokens - 1)
            retrieval_fraction_after_each_turn.append(environment.retrieval_fraction)

        completion = messages[len(row["prompt"]) :]
        return {
            "row": row,
            "completion": completion,
            "full_token_ids": full_token_ids,
            "action_mask": action_mask,
            "turn_boundary_action_indices": turn_boundary_action_indices,
            "retrieval_fraction_after_each_turn": retrieval_fraction_after_each_turn,
        }
