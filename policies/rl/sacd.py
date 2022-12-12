import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import CategoricalPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
import torch.nn.functional as F


class SACD(RLAlgorithmBase):
    name = "sacd"
    continuous_action = False
    use_target_actor = False

    def __init__(
        self,
        entropy_alpha=0.1,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        action_dim=None,
        **kwargs
    ):

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            assert target_entropy is not None
            self.target_entropy = float(target_entropy) * np.log(action_dim)
            self.log_alpha_entropy = torch.zeros(
                1, requires_grad=True, device=ptu.device
            )
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        if type(input_size) == tuple:
            assert len(input_size)==1
            input_size = input_size[0]
        main_actor = CategoricalPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
        return nn.ModuleDict({"main_actor": main_actor})

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        if type(input_size) == tuple:
            assert len(input_size)==1
            input_size = input_size[0]
        qf1 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        return nn.ModuleDict({"main_qf1": qf1, "main_qf2": qf2})

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        action, prob, log_prob = actor(observ, deterministic, return_log_prob)
        return action, prob, log_prob, None

    @staticmethod
    def forward_actor(actor, observ):
        _, probs, log_probs = actor(observ, return_log_prob=True)
        return probs, log_probs  # (T+1, B, dim), (T+1, B, dim)

    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
        **kwargs,
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from current policy,
            if markov_actor:
                new_probs, new_log_probs = actor(next_observs if markov_critic else observs)["main_actor"]
            else:
                # (T+1, B, dim) including reaction to last obs
                new_probs, new_log_probs = actor(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                )["main_actor"]

            if markov_critic:  # (B, A)
                q_dict = critic_target(next_observs)
                next_q1, next_q2 = q_dict["main_qf1"], q_dict["main_qf2"]
            else:
                q_dict = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_probs,
                )  # (T+1, B, A)
                next_q1, next_q2 = q_dict["main_qf1"], q_dict["main_qf2"]

            min_next_q_target = torch.min(next_q1, next_q2)

            min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, A)

            # E_{a'\sim \pi}[Q(h',a')], (T+1, B, 1)
            min_next_q_target = (new_probs * min_next_q_target).sum(
                dim=-1, keepdims=True
            )

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        if markov_critic:
            q_pred_dict = critic(observs)
            q1_pred, q2_pred = q_pred_dict["main_qf1"], q_pred_dict["main_qf2"]
            action = actions.long()  # (B, 1)
            q1_pred = q1_pred.gather(dim=-1, index=action)
            q2_pred = q2_pred.gather(dim=-1, index=action)
            qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
            qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        else:
            # Q(h(t), a(t)) (T, B, 1)
            q_pred_dict = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
            )  # (T, B, A)
            q1_pred, q2_pred = q_pred_dict["main_qf1"], q_pred_dict["main_qf2"]

            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdims=True
            )  # (T, B, 1)
            q1_pred = q1_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
            q2_pred = q2_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)

            # masked Bellman error: masks (T,B,1) ignore the invalid error
            # this is not equal to masks * q1_pred, cuz the denominator in mean()
            # 	should depend on masks > 0.0, not a constant B*T
            assert 'masks' in kwargs
            masks = kwargs['masks']
            num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
            q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
            q_target = q_target * masks
            qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
            qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        return {"main_loss": qf1_loss}, {"main_loss": qf2_loss}

    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
        **kwargs,
    ):
        if markov_actor:
            new_probs, log_probs = actor(observs)["main_actor"]
        else:
            new_probs, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )["main_actor"]  # (T+1, B, A)

        if markov_critic:
            q_dict = critic(observs)
            q1, q2 = q_dict["main_qf1"], q_dict["main_qf2"]
        else:
            q_dict = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_probs,
            )  # (T+1, B, A)
            q1, q2 = q_dict["main_qf1"], q_dict["main_qf2"]
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,A)

        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs
        # E_{a\sim \pi}[Q(h,a)]
        policy_loss = (new_probs * policy_loss).sum(axis=-1, keepdims=True)  # (T+1,B,1)
        if not markov_critic:
            policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        if not markov_actor:
            assert 'masks' in kwargs
            masks = kwargs['masks']
            num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
            policy_loss = (policy_loss * masks).sum() / num_valid

        additional_outputs = {}
        # -> negative entropy (T+1, B, 1)
        additional_outputs['negative_entropy'] = (new_probs * log_probs).sum(axis=-1, keepdims=True)

        return {"main_loss": policy_loss}, additional_outputs

    def update_others(self, additional_outputs):
        assert 'negative_entropy' in additional_outputs
        current_log_probs = additional_outputs['negative_entropy'].mean().item()
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()

        return {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}
