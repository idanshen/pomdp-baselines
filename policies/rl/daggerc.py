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
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
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
        qfs = nn.ModuleDict({"main_qf1": main_qf1, "main_qf2": main_qf2})
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
        return {"actor": ["main"], "critic": ["main"]}
