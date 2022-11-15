"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from policies.models import *
from policies.models.markovian_actor import Actor_Markovian
from policies.models.markovian_critic import Critic_Markovian
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu


class ModelFreeOffPolicy_MLP(nn.Module):
    """
    standard off-policy Markovian Policy using MLP
    including TD3 and SAC
    NOTE: it can only solve MDP problem, not POMDPs
    """

    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        algo_name,
        dqn_layers,
        policy_layers,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # pixel obs
        image_encoder_fn=lambda: None,
        teacher_dir=None,
        state_dim=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name],
                                             teacher_dir=teacher_dir,
                                             action_dim=action_dim,
                                             state_dim=self.state_dim)

        # Markov q networks
        self.critic = Critic_Markovian(
            obs_dim=obs_dim,
            dqn_layers=dqn_layers,
            action_dim=action_dim,
            algo=self.algo,
            image_encoder=image_encoder_fn())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        # target networks
        self.critic_target = copy.deepcopy(self.critic)

        # Markov Actor.
        self.policy = Actor_Markovian(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_layers=policy_layers,
            algo=self.algo,
            image_encoder=image_encoder_fn()
        )
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        # target network
        self.policy_target = copy.deepcopy(self.policy)

    @torch.no_grad()
    def act(self, obs, deterministic=False, return_log_prob=False):
        return self.policy.act(
            obs=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

    def update(self, batch):
        observs, next_observs = batch["obs"], batch["obs2"]  # (B, dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]  # (B, dim)
        teacher_actions = batch["teacher_act"]
        states, next_states = batch["states"], batch["states2"]  # (B, dim)

        ### 1. Critic loss
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.policy,
            actor_target=self.policy_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            next_observs=next_observs,
            states=states,
            next_states=next_states
        )

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        # update q networks
        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

        # soft update
        self.soft_target_update()

        ### 2. Actor loss
        policy_loss, log_probs = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.policy,
            actor_target=self.policy_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            teacher_actions=teacher_actions,
            states=states
        )
        policy_loss = policy_loss.mean()

        # update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

        # update others like alpha
        if log_probs is not None:
            current_log_probs = log_probs.mean().item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)
