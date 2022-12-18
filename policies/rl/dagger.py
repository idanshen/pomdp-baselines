import glob
from ruamel.yaml import YAML
from utils import helpers as utl
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import CategoricalPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu


class DAgger(RLAlgorithmBase):
    name = "DAgger"
    continuous_action = False
    use_target_actor = False

    def __init__(
        self,
        loss_type="L2",
        **kwargs,
    ):
        self.loss_type = loss_type

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
            new_probs, log_probs = actor(observs)["main_actor"]
        else:
            new_probs, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )["main_actor"]  # (T+1, B, A)

        if self.loss_type == "L2": # Imitation using L2 norm
            if markov_actor:
                policy_loss = torch.norm(new_probs - torch.exp(teacher_log_probs), dim=1).unsqueeze(dim=1)
            else:
                policy_loss = torch.norm(new_probs[:-1] - torch.exp(teacher_log_probs[:-1]), dim=2).unsqueeze(dim=2)
        elif self.loss_type == "CE": # Imitation using L2 norm
            if markov_actor:
                policy_loss = -torch.sum(torch.exp(teacher_log_probs) * torch.log(new_probs), dim=1).unsqueeze(dim=1)
            else:
                policy_loss = -torch.sum(torch.exp(teacher_log_probs)[:-1] * torch.log(new_probs)[:-1], dim=2).unsqueeze(dim=2)
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
            additional_outputs['negative_entropy'] = (new_probs * log_probs).sum(axis=-1, keepdims=True)
            additional_outputs['accuracy'] = (torch.max(new_probs, dim=-1)[1] == torch.max(torch.exp(teacher_log_probs), dim=-1)[1]).unsqueeze(dim=-1)

        return {"main_loss": policy_loss}, additional_outputs

    def update_others(self, additional_outputs, **kwargs):
        assert 'negative_entropy' in additional_outputs
        current_log_probs = additional_outputs['negative_entropy'].mean().item()
        accuracy = additional_outputs['accuracy'].cpu().numpy().mean().item()
        return {"policy_entropy": -current_log_probs, 'accuracy': accuracy}
