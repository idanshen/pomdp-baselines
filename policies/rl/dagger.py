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
        entropy_alpha=0.1,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        action_dim=None,
        state_dim=None,
        teacher_dir=None,
    ):
        self.loss_type = "L2"

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return CategoricalPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

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
        states=None
    ):
        return (torch.zeros_like(rewards)[1:], torch.zeros_like(rewards)[1:]), torch.zeros_like(rewards, requires_grad=True)[1:]

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
        teacher_actions=None,
    ):
        if markov_actor:
            new_probs, log_probs = actor(observs)
        else:
            new_probs, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)

        if self.loss_type == "L2": # Imitation using L2 norm
            if markov_actor:
                policy_loss = torch.norm(new_probs - torch.exp(teacher_actions), dim=1).unsqueeze(dim=1)
            else:
                policy_loss = torch.norm(new_probs[:-1] - torch.exp(teacher_actions[:-1]), dim=2).unsqueeze(dim=2)
        else:
            raise NotImplementedError("Currently support only L2 loss")

        # -> negative entropy (T+1, B, 1)
        with torch.no_grad():
            log_probs_est = (new_probs * log_probs).sum(axis=-1, keepdims=True)

        return policy_loss, log_probs_est

    def update_others(self, current_log_probs):
        return {"policy_entropy": -current_log_probs}
        # return {"imitation accuracy": current_log_probs}
