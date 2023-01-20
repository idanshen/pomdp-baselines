import glob
import math

from ruamel.yaml import YAML
from utils import helpers as utl
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import CategoricalPolicy, TanhGaussianPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
import torch.nn.functional as F


class DAggerc(RLAlgorithmBase):
    name = "DAggerc"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        loss_type="L2",
        **kwargs,
    ):
        super().__init__()
        self.loss_type = loss_type
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
        return nn.ModuleDict({"main_actor": main_actor})

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=1, action_dim=None):
        assert action_dim is not None
        if type(obs_dim) == tuple:
            assert len(obs_dim) == 1
            obs_dim = obs_dim[0]
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        main_qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        main_qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        aux_v1 = FlattenMlp(
            input_size=obs_dim, output_size=1, hidden_sizes=hidden_sizes
        )
        aux_v2 = FlattenMlp(
            input_size=obs_dim, output_size=1, hidden_sizes=hidden_sizes
        )
        # qfs = nn.ModuleDict({"main_qf1": main_qf1, "main_qf2": main_qf2})
        qfs = nn.ModuleDict({"main_qf1": main_qf1, "main_qf2": main_qf2, "aux_qf1": aux_v1, "aux_qf2": aux_v2})
        return qfs

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        action, mean, log_std, log_prob = actor(observ, False, deterministic, return_log_prob)
        return action, mean, log_std, log_prob

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, mean, log_std, log_probs = actor(observ, return_log_prob=True)
        return new_actions, mean, log_std, log_probs  # (T+1, B, dim), (T+1, B, dim)

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
        states=None,
        **kwargs
    ):
        # with torch.no_grad():
        #     # first next_actions from current policy,
        #     if markov_actor:
        #         new_actions, new_mean, new_log_std, new_log_probs = actor["main"](
        #             next_observs if markov_critic else observs)
        #     else:
        #         # (T+1, B, dim) including reaction to last obs
        #         new_actions, new_mean, new_log_std, new_log_probs = actor["main"](
        #             prev_actions=actions,
        #             rewards=rewards,
        #             observs=next_observs if markov_critic else observs,
        #         )
        #
        #
        #     if markov_critic:  # (B, 1)
        #         next_q1, next_q2 = critic_target["main"](next_observs, new_actions)
        #     else:
        #         next_q1, next_q2 = critic_target["main"](
        #             prev_actions=actions,
        #             rewards=rewards,
        #             observs=observs,
        #             current_actions=new_actions,
        #         )  # (T+1, B, 1)
        #
        #     min_next_q_target = torch.min(next_q1, next_q2)
        #
        #     # q_target: (T, B, 1)
        #     q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
        #     if not markov_critic:
        #         q_target = q_target[1:]  # (T, B, 1)
        #
        #     # And also current_new_actions for the value function
        #     if markov_actor:
        #         new_curr_actions, _, _, _ = actor["main"](observs)
        #         next_q1, next_q2 = critic_target["main"](observs, new_curr_actions)
        #     else:
        #         raise NotImplementedError
        #     min_curr_q_target = torch.min(next_q1, next_q2)
        #
        # if markov_critic:
        #     q1_pred, q2_pred = critic["main"](observs, actions)
        #     v1_pred, v2_pred = critic["aux"](observs)
        #     qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        #     qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error
        #     v1_loss = F.mse_loss(v1_pred, min_curr_q_target)  # TD error
        #     v2_loss = F.mse_loss(v2_pred, min_curr_q_target)  # TD error
        #
        # else:
        #     # Q(h(t), a(t)) (T, B, 1)
        #     q1_pred, q2_pred = critic["main"](
        #         prev_actions=actions,
        #         rewards=rewards,
        #         observs=observs,
        #         current_actions=actions[1:],
        #     )  # (T, B, 1)
        #
        #     # masked Bellman error: masks (T,B,1) ignore the invalid error
        #     # this is not equal to masks * q1_pred, cuz the denominator in mean()
        #     # 	should depend on masks > 0.0, not a constant B*T
        #     assert 'masks' in kwargs
        #     masks = kwargs['masks']
        #     num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
        #     q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        #     q_target = q_target * masks
        #     qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        #     qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error
        #
        # return {"main_loss": qf1_loss, "aux_loss": v1_loss}, {"main_loss": qf2_loss, "aux_loss": v2_loss}
        return {"main_loss": torch.zeros(1, requires_grad=True)}, {"main_loss": torch.zeros(1, requires_grad=True)}


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
        states=None,
        teacher_log_probs=None,
        **kwargs
    ):
        if markov_actor:
            new_actions, mean, log_std, log_probs = actor["main"](observs)
        else:
            new_actions, mean, log_std, log_probs = actor["main"](
                prev_actions=actions, rewards=rewards, observs=observs
            ) # (T+1, B, A)

        if self.loss_type == "L2":  # Imitation using L2 norm
            if markov_actor:
                policy_loss = torch.norm(new_actions - teacher_log_probs, dim=1).unsqueeze(dim=1)
            else:
                policy_loss = torch.norm(new_actions[:-1] - teacher_log_probs[:-1], dim=2).unsqueeze(dim=2)
        elif self.loss_type == "CE":  # Imitation using L2 norm
            pre_tanh_teacher_actions = torch.atanh(teacher_log_probs)

            var = torch.exp(log_std) ** 2
            log_prob_teacher_actions = (
                    -((pre_tanh_teacher_actions - mean) ** 2) / (2 * var)
                    - log_std
                    - math.log(math.sqrt(2 * np.pi))
                    - torch.log(1 - teacher_log_probs * teacher_log_probs + self.epsilon)
            )

            if markov_actor:
                policy_loss = -log_prob_teacher_actions
            else:
                policy_loss = -log_prob_teacher_actions[:-1]
        else:
            raise NotImplementedError("Currently support only L2 loss")

        if not markov_actor:
            assert 'masks' in kwargs
            masks = kwargs['masks']
            num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
            policy_loss = (policy_loss * masks).sum() / num_valid

        additional_outputs = {}
        # -> negative entropy (T+1, B, 1)
        with torch.no_grad():
            # additional_outputs['negative_entropy'] = (new_probs * log_probs).sum(axis=-1, keepdims=True)
            additional_outputs['accuracy'] = torch.abs(teacher_log_probs - mean) < 0.01

        return {"main_loss": policy_loss}, additional_outputs

    def update_others(self, additional_outputs, **kwargs):
        # assert 'negative_entropy' in additional_outputs
        # current_log_probs = additional_outputs['negative_entropy'].mean().item()
        accuracy = additional_outputs['accuracy'].cpu().numpy().mean().item()
        return {'accuracy': accuracy}

    @property
    def model_keys(self):
        return {"actor": ["main"], "critic": ["main", "aux"]}
        # return {"actor": ["main"], "critic": ["main"]}