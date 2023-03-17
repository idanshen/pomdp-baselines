import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.distributions import Categorical
from torch.optim import Adam

import torchkit.pytorch_utils as ptu
from policies.models.actor import CategoricalPolicy, TanhGaussianPolicy
from torchkit.networks import FlattenMlp, DoubleHeadFlattenMlp
from utils import helpers as utl
from .base import RLAlgorithmBase


class ELFc(RLAlgorithmBase):
    name = "elfc"
    continuous_action = True
    use_target_actor = False

    def __init__(
            self,
            min_v=0.0,
            action_dim=None,
            obs_dim=None,
            imitation_policy_dir=None,
            **kwargs
    ):
        super().__init__()
        self.action_dim = action_dim
        self.imitation_policy, self.imitation_critic = utl.load_teacher(imitation_policy_dir, state_dim=obs_dim, act_dim=action_dim)
        self.min_v = min_v
        self.alpha_entropy = 0.01

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

    @torch.no_grad()
    def calculate_follower_values(self, markov_actor, markov_critic, actor, critic, observs, next_observs, actions, rewards, dones):
        if "markovian" in str(type(self.imitation_critic["aux"])):
            curr_obs_v1, curr_obs_v2 = self.imitation_critic["aux"](observs)
            curr_obs_values = torch.min(curr_obs_v1, curr_obs_v2)

            next_obs_v1, next_obs_v2 = self.imitation_critic["aux"](next_observs)
            next_obs_values = torch.min(next_obs_v1, next_obs_v2)
            next_obs_values = next_obs_values * (1.0 - dones)  # Last state get only -v(s_t) without +v(s_t+1)
        else:
            raise NotImplementedError

        return curr_obs_values, next_obs_values

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
            next_states=None,
            teacher_log_probs=None,
            teacher_next_log_probs=None,
            **kwargs
    ):
        with torch.no_grad():
            # first next_actions from current policy,
            if markov_actor:
                new_actions, new_mean, new_log_std, new_log_probs = actor["main"](
                    next_observs if markov_critic else observs)
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

            values_curr_obs, values_next_obs = self.calculate_follower_values(markov_actor=markov_actor,
                                                                              markov_critic=markov_critic,
                                                                              actor=actor,
                                                                              critic=critic,
                                                                              observs=observs,
                                                                              next_observs=next_observs if markov_critic else observs,
                                                                              actions=actions,
                                                                              rewards=rewards,
                                                                              dones=dones)
            if not markov_critic:
                _, batch_size, _ = values_curr_obs.shape
                values_curr_obs = torch.cat((ptu.zeros((1, batch_size, 1)).float(), values_curr_obs[1:]), dim=0)
            rewards_aug = rewards + gamma * values_next_obs - values_curr_obs

            # q_target: (T, B, 1)
            q_target = rewards_aug + (1.0 - dones) * gamma * min_next_q_target  # next q
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
            next_observs=None,
            actions=None,
            rewards=None,
            states=None,
            teacher_log_probs=None,
            dones=None,
            **kwargs
    ):
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

        # CE_loss = torch.norm(new_actions - teacher_log_probs, dim=1).unsqueeze(dim=1)  # (T+1,B,1)

        # values_curr_obs, values_next_obs = self.calculate_follower_values(markov_actor=markov_actor,
        #                                                                   markov_critic=markov_critic,
        #                                                                   actor=actor,
        #                                                                   critic=critic,
        #                                                                   observs=observs,
        #                                                                   next_observs=next_observs,
        #                                                                   actions=actions,
        #                                                                   rewards=rewards,
        #                                                                   dones=dones)
        # mask = torch.zeros_like(values_curr_obs)
        # mask[values_curr_obs > self.min_v] = 1.0

        if not markov_critic:
            env_loss = env_loss[:-1]  # (T,B,1) remove the last obs
            # CE_loss = CE_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = env_loss #+ mask * CE_loss

        if not markov_actor:
            assert 'masks' in kwargs
            masks = kwargs['masks']
            num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
            policy_loss = (policy_loss * masks).sum() / num_valid

        additional_outputs = {}
        # -> negative entropy (T+1, B, 1)
        additional_outputs['negative_entropy'] = log_probs
        additional_outputs['std'] = torch.exp(log_std)
        additional_outputs['abs_mean'] = torch.abs(mean)

        return {"main_loss": policy_loss}, additional_outputs

    def update_others(self, additional_outputs, **kwargs):
        assert 'negative_entropy' in additional_outputs
        current_entropy = -additional_outputs['negative_entropy'].mean().item()
        current_std = additional_outputs['std'].mean().item()
        current_abs_mean = additional_outputs['abs_mean'].mean().item()
        output_dict = {"mean_std": current_std,
                       "average_abs_mean": current_abs_mean,
                       "policy_entropy": current_entropy,
                       }
        return output_dict
