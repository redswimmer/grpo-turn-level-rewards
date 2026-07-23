"""train_ppo.py: custom multi-turn PPO trainer (MTPPOTrainer) for ppo/mt_ppo conditions.

Built directly on transformers.Trainer, not GRPOTrainer/PPOTrainer -- TRL's PPOTrainer has no
multi-turn tool-calling support (confirmed fresh against the installed 1.7.1 and upstream's
dev branch, re-verified 2026-07-23; see
docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md). Reuses SearchEnv/rewards.py/data.py
unmodified. See CLAUDE.md's Goal section and docs/phase-7-mt-ppo.md for the full design.
"""

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import trackio
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response

from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import TURN_REWARD_SCALE, format_reward, outcome_reward

Condition = Literal["ppo", "mt_ppo"]

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

_SAMPLE_COMPLETION_INTERVAL = 10


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

            # policy.device is a real PreTrainedModel attribute, but ty can't see through the
            # loose Trainer base-class type annotation on self.model -- same root cause as the
            # unresolved-attribute suppressions in create_optimizer above.
            input_ids = torch.tensor([prompt_token_ids], device=policy.device)  # ty: ignore[invalid-argument-type]
            with torch.no_grad():
                # policy.generate is a real PreTrainedModel method, but ty can't see it through
                # the loose Trainer base-class type annotation -- policy is provably an
                # AutoModelForCausalLM instance at runtime, always has .generate.
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

    def _forward_logprobs_and_values(
        self, full_token_ids: list[int], action_mask: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass over one episode's full token sequence.

        Returns (action_logprobs, action_values), both 1-D tensors of length sum(action_mask) --
        one entry per action (policy-generated) token, in the same order as
        _rollout_episode's turn_boundary_action_indices.

        Called twice per episode per PPO update: once under torch.no_grad() right after rollout
        (the frozen "old" reference for the clip ratio, GAE, and the KL term), and once per PPO
        inner epoch WITH grad (the "new" values that get backpropagated).
        """
        # self.model.policy resolves through Trainer's loose base-class type (nn.Module | None),
        # so ty can't see .device as a real attribute -- same root cause as the
        # unresolved-attribute suppressions already used in create_optimizer and
        # _rollout_episode above. Safe: self.model.policy is provably a real PreTrainedModel
        # instance at runtime, always has a genuine .device.
        device = self.model.policy.device  # ty: ignore[unresolved-attribute]
        # device's type is therefore unresolved/ambiguous to ty (see immediately above), so it
        # can't confirm device is a valid torch.device-compatible value for the `device=` kwarg
        # on either torch.tensor(...) call below -- the same unresolved-attribute-propagates-
        # into-invalid-argument-type chain _rollout_episode's input_ids construction already
        # explains for policy.device. Safe for the same reason: at runtime device is always a
        # genuine torch.device.
        input_ids = torch.tensor([full_token_ids], device=device)  # ty: ignore[invalid-argument-type]
        mask = torch.tensor(action_mask, device=device, dtype=torch.bool)  # ty: ignore[invalid-argument-type]

        # self.model.policy is untyped/ambiguous to ty (see above), so calling it as
        # `self.model.policy(...)` looks like a call-non-callable (ty can't confirm it's a
        # callable nn.Module) with an unresolved-attribute layered on top (ty can't see
        # `.policy` on self.model's loose base type either). Safe: self.model.policy is provably
        # an AutoModelForCausalLM instance at runtime, always callable.
        policy_logits = self.model.policy(input_ids=input_ids).logits[0]  # ty: ignore[call-non-callable, unresolved-attribute]  # [seq_len, vocab]
        # logits[t] predicts token[t+1]; gather log-prob of the actual next token at each
        # position, then select the ones landing on action tokens (shifted by one: an action
        # token at absolute position t was predicted by logits at position t-1).
        log_probs_all = torch.log_softmax(policy_logits[:-1], dim=-1)
        next_tokens = input_ids[0, 1:]
        token_logprobs = log_probs_all.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
        action_indices = mask.nonzero(as_tuple=True)[0]
        action_logprobs = token_logprobs[action_indices - 1]

        # self.model.critic is untyped/ambiguous to ty for the same reason as self.model.policy
        # above -- `.model` resolves to an unresolved-attribute, and calling the result looks
        # like a call-non-callable. Safe: self.model.critic.model is provably the real
        # transformer backbone (a PreTrainedModel) at runtime, always callable.
        critic_hidden = self.model.critic.model(input_ids=input_ids).last_hidden_state  # ty: ignore[call-non-callable, unresolved-attribute]
        # Same root cause one line up: self.model.critic.score is provably a real nn.Linear
        # value head at runtime, always callable, but ty can't see `.score` through self.model's
        # loose base type either.
        critic_values = self.model.critic.score(critic_hidden).squeeze(-1)[0]  # ty: ignore[call-non-callable, unresolved-attribute]  # [seq_len]
        action_values = critic_values[action_indices]

        return action_logprobs, action_values

    def _collect_batch(self, rows: list[dict]) -> list[dict]:
        """Roll out one episode per row, score it, and compute its frozen GAE inputs.

        Not unit-tested -- calls _rollout_episode (real model/tool-calls) and
        _forward_logprobs_and_values (real forward pass) for each row. Validated by the live
        smoke test (Task 11).
        """
        episodes = []
        for row in rows:
            rollout = self._rollout_episode(row)

            with torch.no_grad():
                old_logprobs, old_values = self._forward_logprobs_and_values(
                    rollout["full_token_ids"], rollout["action_mask"]
                )

            completion = rollout["completion"]
            format_r = format_reward([completion])[0]
            outcome_r = outcome_reward([completion], [row["golden_answers"]])[0]
            format_and_outcome_reward = format_r + outcome_r

            retrieval_fraction = (
                rollout["retrieval_fraction_after_each_turn"][-1]
                if rollout["retrieval_fraction_after_each_turn"]
                else 0.0
            )

            per_token_rewards = place_turn_rewards(
                num_tokens=len(old_values),
                turn_boundary_token_indices=rollout["turn_boundary_action_indices"],
                retrieval_fraction_after_each_turn=rollout["retrieval_fraction_after_each_turn"],
                format_and_outcome_reward=format_and_outcome_reward,
                condition=self.condition,
            )
            advantages = compute_gae(
                rewards=per_token_rewards,
                values=old_values.tolist(),
                # self.args.gamma and self.args.gae_lambda are real fields on this file's
                # MTPPOConfig (set in build_ppo_config), but Trainer's base type stub only
                # knows self.args as the looser `TrainingArguments` -- same ty-can't-see-
                # through-Trainer's-base-types root cause already noted on self.args.n_max in
                # _rollout_episode above. Safe to ignore here for the same reason on both of
                # these adjacent kwargs.
                gamma=self.args.gamma,  # ty: ignore[unresolved-attribute]
                lam=self.args.gae_lambda,  # ty: ignore[unresolved-attribute]
            )
            returns = [a + v for a, v in zip(advantages, old_values.tolist(), strict=True)]

            episodes.append(
                {
                    "full_token_ids": rollout["full_token_ids"],
                    "action_mask": rollout["action_mask"],
                    "old_logprobs": old_logprobs,
                    "old_values": old_values,
                    "advantages": torch.tensor(advantages, device=old_values.device),
                    "returns": torch.tensor(returns, device=old_values.device),
                    "format_and_outcome_reward": format_and_outcome_reward,
                    "retrieval_fraction": retrieval_fraction,
                    "question": row["question"],
                    "completion": completion,
                }
            )
        return episodes

    def _ppo_update(self, episodes: list[dict]) -> dict[str, float]:
        """Run args.num_ppo_epochs inner passes over the collected batch, gradient-accumulated
        across all episodes in the batch, one optimizer step per inner epoch. Matches this
        repo's existing train.py precedent of per-episode (batch-of-1) forward/backward passes
        with gradient accumulation, avoiding any padding/attention-mask complexity -- consistent
        with the single-RTX-4090, 0.8B-model memory profile the rest of this repo already
        established.
        """
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
        num_updates = 0
        # self.args.num_ppo_epochs is a real field on this file's MTPPOConfig, but Trainer's
        # base type stub only knows self.args as the looser `TrainingArguments` -- same
        # ty-can't-see-through-Trainer's-base-types root cause already noted throughout this
        # class (see create_optimizer, _rollout_episode, _collect_batch above).
        for _epoch in range(self.args.num_ppo_epochs):  # ty: ignore[unresolved-attribute]
            # self.optimizer is declared Optional (`Optimizer | None | Unknown`) on Trainer's
            # base class, since Trainer only populates it once create_optimizer() has actually
            # been called -- but train() always assigns a real optimizer
            # (self.optimizer = self.create_optimizer()) before ever invoking _ppo_update, so it
            # is provably non-None here at runtime.
            self.optimizer.zero_grad()  # ty: ignore[unresolved-attribute]
            for episode in episodes:
                new_logprobs, new_values = self._forward_logprobs_and_values(
                    episode["full_token_ids"], episode["action_mask"]
                )
                action_mask = torch.ones_like(new_logprobs)
                loss_dict = compute_ppo_loss(
                    new_logprobs=new_logprobs,
                    old_logprobs=episode["old_logprobs"],
                    advantages=episode["advantages"],
                    returns=episode["returns"],
                    new_values=new_values,
                    action_mask=action_mask,
                    # self.args.clip_eps, self.args.kl_beta, and self.args.value_loss_coef are
                    # real fields on this file's MTPPOConfig -- same ty-can't-see-through-
                    # Trainer's-base-types root cause as self.args.num_ppo_epochs above; these
                    # three are genuinely back-to-back kwargs sharing that identical cause.
                    clip_eps=self.args.clip_eps,  # ty: ignore[unresolved-attribute]
                    kl_beta=self.args.kl_beta,  # ty: ignore[unresolved-attribute]
                    value_loss_coef=self.args.value_loss_coef,  # ty: ignore[unresolved-attribute]
                )
                (loss_dict["loss"] / len(episodes)).backward()
                for key in totals:
                    totals[key] += loss_dict[key].item()
                num_updates += 1
            # self.optimizer is Optional on Trainer's base class for the same reason noted
            # above this method's self.optimizer.zero_grad() call; still provably a real
            # Optimizer at this point in the loop.
            self.optimizer.step()  # ty: ignore[unresolved-attribute]
        return {key: value / num_updates for key, value in totals.items()}

    # train(self) intentionally narrows Trainer.train's full signature
    # (resume_from_checkpoint, trial, ignore_keys_for_eval -> TrainOutput) down to train(self)
    # -> None: PPO's collect-then-multi-epoch-update outer loop replaces Trainer's generic
    # dataloader/training_step machinery those parameters exist to support (see this method's
    # own docstring), and no call site in this repo (build_ppo_trainer, the live smoke test)
    # ever needs to pass them.
    def train(self) -> None:  # ty: ignore[invalid-method-override]
        """Overrides Trainer.train() entirely: PPO's collect-then-multi-epoch-update structure
        doesn't fit Trainer's default single-pass-per-batch loop, so this owns the whole outer
        loop instead of relying on get_train_dataloader()/training_step().

        Writes two diagnostic artifacts under self.args.output_dir, so a run can be inspected
        after the fact without rerunning it:
          - train_log.jsonl: one JSON line per step with every metric plus a per-episode
            breakdown (question, reward, retrieval_fraction, action-token count) -- enough detail
            to diagnose a specific step or a specific episode's reward after training has already
            finished, not just an aggregate curve.
          - sample_completions.log: one full example transcript appended every
            _SAMPLE_COMPLETION_INTERVAL steps (plain text), mirroring this repo's existing
            train.py convention of log_completions=True for GRPO -- lets a human spot-check real
            model output during a long run without re-running anything.
        Also prints a one-line progress summary to stdout each step (step/max_steps, key metrics,
        elapsed and estimated-remaining wall-clock) so a long run's progress is visible without
        having trackio's dashboard open.

        set_seed(self.args.seed) is called here explicitly because this override replaces
        Trainer.train() entirely -- the base class's own seeding call never runs, so without this,
        two runs with the same --seed would silently stop reproducing the same rollouts (sampling
        in _rollout_episode uses torch's global RNG).
        """
        set_seed(self.args.seed)
        self.optimizer = self.create_optimizer()
        # self.train_dataset is typed by Trainer's base class as
        # `torch.utils.data.Dataset | datasets.arrow_dataset.Dataset | None`, and
        # torch.utils.data.Dataset's stub doesn't declare __iter__, so ty can't confirm it's
        # Iterable -- but build_ppo_trainer (Task 11) always constructs this trainer with a real
        # datasets.Dataset, which genuinely is iterable at runtime.
        rows = list(self.train_dataset)  # ty: ignore[invalid-argument-type]
        row_cycle = itertools.cycle(rows)
        trackio.init(project=self.args.project, name=self.args.run_name)

        # self.args.output_dir is typed `str | None` on TrainingArguments (None only if a
        # caller explicitly passed output_dir=None), but build_ppo_config always sets a real
        # output_dir string, so this is never actually None at runtime.
        output_dir = Path(self.args.output_dir)  # ty: ignore[invalid-argument-type]
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.jsonl"
        sample_completions_path = output_dir / "sample_completions.log"

        run_start = time.monotonic()
        for step in range(self.args.max_steps):
            step_start = time.monotonic()
            # self.args.num_rollouts_per_step is a real field on this file's MTPPOConfig, but
            # Trainer's base type stub only knows self.args as the looser `TrainingArguments` --
            # same ty-can't-see-through-Trainer's-base-types root cause already noted throughout
            # this class (see create_optimizer, _rollout_episode, _collect_batch, _ppo_update
            # above).
            batch_rows = [
                next(row_cycle)
                for _ in range(self.args.num_rollouts_per_step)  # ty: ignore[unresolved-attribute]
            ]
            episodes = self._collect_batch(batch_rows)
            update_metrics = self._ppo_update(episodes)

            mean_reward = sum(e["format_and_outcome_reward"] for e in episodes) / len(episodes)
            mean_retrieval_fraction = sum(e["retrieval_fraction"] for e in episodes) / len(episodes)
            total_elapsed = time.monotonic() - run_start
            steps_remaining = self.args.max_steps - (step + 1)
            eta_seconds = (total_elapsed / (step + 1)) * steps_remaining

            metrics = {
                "step": step,
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "kl": update_metrics["kl"],
                "reward": mean_reward,
                "retrieval_fraction": mean_retrieval_fraction,
            }
            trackio.log(metrics)

            log_record = dict(metrics)
            log_record["step_elapsed_seconds"] = time.monotonic() - step_start
            log_record["total_elapsed_seconds"] = total_elapsed
            log_record["eta_seconds"] = eta_seconds
            log_record["episodes"] = [
                {
                    "question": episode["question"],
                    "format_and_outcome_reward": episode["format_and_outcome_reward"],
                    "retrieval_fraction": episode["retrieval_fraction"],
                    "num_action_tokens": len(episode["old_values"]),
                }
                for episode in episodes
            ]
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(log_record) + "\n")

            if step == 0 or (step + 1) % _SAMPLE_COMPLETION_INTERVAL == 0:
                sample_episode = episodes[0]
                with sample_completions_path.open("a") as sample_file:
                    sample_file.write(
                        f"=== step {step + 1} | reward="
                        f"{sample_episode['format_and_outcome_reward']:.3f} | retrieval_fraction="
                        f"{sample_episode['retrieval_fraction']:.3f} ===\n"
                    )
                    sample_file.write(f"question: {sample_episode['question']}\n")
                    for message in sample_episode["completion"]:
                        sample_file.write(f"[{message.get('role')}] {message.get('content')}\n")
                    sample_file.write("\n")

            print(
                f"step {step + 1}/{self.args.max_steps} | loss={metrics['loss']:.4f} "
                f"reward={mean_reward:.3f} retrieval_fraction={mean_retrieval_fraction:.3f} "
                f"| elapsed={total_elapsed:.0f}s eta={eta_seconds:.0f}s",
                flush=True,
            )

            self.state.global_step = step + 1

            if (step + 1) % self.args.save_steps == 0 if self.args.save_steps else False:
                self.save_model(f"{self.args.output_dir}/checkpoint-{step + 1}")

        self.save_model(f"{self.args.output_dir}/checkpoint-{self.args.max_steps}")
