import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
import torch.nn.functional as F


class ADVISORc(RLAlgorithmBase):
    name = "advisorc"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        temprature=1.0,
        alpha_entropy=1.0,
        action_dim=None,
        **kwargs
    ):
        super().__init__()
        self.action_dim = action_dim
        self.temprature = temprature
        self.alpha_entropy = alpha_entropy
        self.epsilon = 1e-6

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        if type(input_size) == tuple:
            assert len(input_size)==1
            input_size = input_size[0]
        main_actor = TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
        aux_actor = TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
        return nn.ModuleDict({"main_actor": main_actor, "aux_actor": aux_actor})

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if type(obs_dim) == tuple:
            assert len(obs_dim)==1
            obs_dim = obs_dim[0]
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return nn.ModuleDict({"main_qf1": qf1, "main_qf2": qf2})

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        action, mean, log_std, log_prob = actor(observ, False, deterministic, return_log_prob)
        return action, mean, log_std, log_prob

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, mean, log_std, log_probs = actor(observ, return_log_prob=True)
        return new_actions, mean, log_std, log_probs  # (T+1, B, dim), (T+1, B, 1)

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
                new_actions, new_mean, new_log_std, new_log_probs = actor["main"](next_observs if markov_critic else observs)
            else:
                # (T+1, B, dim) including reaction to last obs
                new_actions, new_mean, new_log_std, new_log_probs = actor["main"](
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                )

            if markov_critic:  # (B, 1)
                next_q1, next_q2 = critic_target["main"](next_observs, new_actions)
            else:
                next_q1, next_q2 = critic_target["main"](
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_actions,
                )  # (T+1, B, 1)

            min_next_q_target = torch.min(next_q1, next_q2)
            # min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, 1)

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        if markov_critic:
            q1_pred, q2_pred = critic["main"](observs, actions)
            qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
            qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        else:
            # Q(h(t), a(t)) (T, B, 1)
            q1_pred, q2_pred = critic["main"](
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
            )  # (T, B, 1)

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
        teacher_log_probs=None,
        **kwargs,
    ):
        # Aux policy loss
        if markov_actor:
            new_aux_actions, aux_mean, _, _ = actor["aux"](observs)
        else:
            new_aux_actions, aux_mean, _, _ = actor["aux"](
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)
        policy_loss_aux = torch.norm(new_aux_actions - teacher_log_probs, dim=1).unsqueeze(dim=1)

        # Main policy loss
        if markov_actor:
            new_actions, mean, log_std, log_probs = actor["main"](observs)
        else:
            new_actions, mean, log_std, log_probs = actor["main"](
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)

        if markov_critic:
            q1, q2 = critic["main"](observs, new_actions)
        else:
            q1, q2 = critic["main"](
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_actions,
            )  # (T+1, B, 1)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        env_loss = -min_q_new_actions
        env_loss += self.alpha_entropy * log_probs
        CE_loss = torch.norm(new_actions - teacher_log_probs, dim=1).unsqueeze(dim=1)
        dis = torch.norm(aux_mean.detach() - teacher_log_probs, dim=1).unsqueeze(dim=1)
        coefficient = torch.exp(-self.temprature * dis)
        policy_loss_main = coefficient * CE_loss + (1.0 - coefficient) * env_loss

        if not markov_critic:
            policy_loss_aux = policy_loss_aux[:-1]  # (T,B,1) remove the last obs
            policy_loss_main = policy_loss_main[:-1]  # (T,B,1) remove the last obs

        policy_loss = policy_loss_aux + policy_loss_main

        if not markov_actor:
            assert 'masks' in kwargs
            masks = kwargs['masks']
            num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
            policy_loss = (policy_loss * masks).sum() / num_valid

        additional_outputs = {}
        # -> negative entropy (T+1, B, 1)
        additional_outputs['negative_entropy'] = log_probs
        pre_tanh_teacher_actions = torch.atanh(teacher_log_probs)
        var = torch.exp(log_std) ** 2
        log_prob_teacher_actions = (
                -((pre_tanh_teacher_actions - mean) ** 2) / (2 * var)
                - log_std
                - math.log(math.sqrt(2 * np.pi))
                - torch.log(1 - teacher_log_probs * teacher_log_probs + self.epsilon)
        )
        additional_outputs['negative_cross_entropy'] = log_prob_teacher_actions
        additional_outputs['std'] = torch.exp(log_std)
        additional_outputs['abs_mean'] = torch.abs(mean)
        additional_outputs['average_coeff'] = coefficient

        return {"main_loss": policy_loss}, additional_outputs

    def update_others(self, additional_outputs, **kwargs):
        assert 'negative_entropy' in additional_outputs
        current_log_probs = additional_outputs['negative_entropy'].mean().item()
        current_std = additional_outputs['std'].mean().item()
        current_abs_mean = additional_outputs['abs_mean'].mean().item()
        current_cross_entropy = -additional_outputs['negative_cross_entropy'].mean().item()
        current_average_coeff = additional_outputs['average_coeff'].mean().item()

        return {"policy_entropy": -current_log_probs, "cross_entropy": current_cross_entropy, "mean_std": current_std, "average_abs_mean": current_abs_mean, "average_coeff": current_average_coeff}

    def get_acting_policy_key(self):
        return self.current_policy

    @property
    def model_keys(self):
        return {"actor": ["main", "aux"], "critic": ["main"]}