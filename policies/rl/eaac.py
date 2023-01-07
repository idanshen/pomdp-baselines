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


class EAAC(RLAlgorithmBase):
    name = "eaac"
    continuous_action = True
    use_target_actor = False

    def __init__(
            self,
            initial_coefficient=1.0,
            coefficient_tuning="Fixed",
            target_coefficient=None,
            coefficient_lr=3e-4,
            action_dim=None,
            state_dim=None,
            teacher_dir=None,
            min_coefficient=0.01,
            max_coefficient=3.0,
            split_q=False,
    ):
        super().__init__()
        assert coefficient_tuning in ["Fixed", "Target", "EIPO"]
        self.coefficient_tuning = coefficient_tuning
        if self.coefficient_tuning == "EIPO":
            self.obj_est_main = 0.0
            self.obj_est_aux = 0.0
        self.split_q = split_q
        self.action_dim = action_dim
        # self.cross_entropy_sum = 0
        # self.cross_entropy_sumsq = 0
        # self.cross_entropy_count = 0

        self.min_coefficent = min_coefficient
        self.max_coefficent = max_coefficient
        if self.coefficient_tuning == "Target":
            assert target_coefficient is not None
            self.target_coefficient = float(target_coefficient)
            self.log_coefficient = torch.tensor(np.log(initial_coefficient), requires_grad=True, device=ptu.device)
            self.coefficient_optim = Adam([self.log_coefficient], lr=coefficient_lr)
            self.coefficient = self.log_coefficient.exp().detach().item()
        elif self.coefficient_tuning == "Fixed":
            self.coefficient = initial_coefficient
        else:
            self.log_coefficient = torch.tensor(np.log(initial_coefficient), requires_grad=True, device=ptu.device)
            self.coefficient = self.log_coefficient.exp().detach().item()
            self.coefficient_lr = coefficient_lr / 500.0

    def build_actor(self, input_size, action_dim, hidden_sizes, **kwargs):
        if type(input_size) == tuple:
            assert len(input_size) == 1
            input_size = input_size[0]

        main_actor = TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
        actors = nn.ModuleDict({"main_actor": main_actor})
        if self.coefficient_tuning in ["EIPO"]:
            aux_actor = TanhGaussianPolicy(
                obs_dim=input_size,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                **kwargs,
            )
            actors["aux_actor"] = aux_actor
        return actors

    def build_critic(self, hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        assert action_dim is not None
        if type(obs_dim) == tuple:
            assert len(obs_dim) == 1
            obs_dim = obs_dim[0]
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        if self.split_q:
            main_qf1 = DoubleHeadFlattenMlp(
                input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
            )
            main_qf2 = DoubleHeadFlattenMlp(
                input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
            )
        else:
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
            next_states=None,
            teacher_log_probs=None,
            teacher_next_log_probs=None,
            **kwargs
    ):
        main_qf1_loss, main_qf2_loss = self._critic_loss("main",
                                                         markov_actor=markov_actor,
                                                         markov_critic=markov_critic,
                                                         actor=actor,
                                                         actor_target=actor_target,
                                                         critic=critic,
                                                         critic_target=critic_target,
                                                         observs=observs,
                                                         actions=actions,
                                                         rewards=rewards,
                                                         dones=dones,
                                                         gamma=gamma,
                                                         next_observs=next_observs,
                                                         states=states,
                                                         next_states=next_states,
                                                         teacher_log_probs=teacher_log_probs,
                                                         teacher_next_log_probs=teacher_next_log_probs,
                                                         **kwargs)
        if self.split_q:
            loss_dict_q1 = {"main_teacher_loss": main_qf1_loss[0], "main_env_loss": main_qf1_loss[1]}
            loss_dict_q2 = {"main_teacher_loss": main_qf2_loss[0], "main_env_loss": main_qf2_loss[1]}
        else:
            loss_dict_q1 = {"main_loss": main_qf1_loss}
            loss_dict_q2 = {"main_loss": main_qf2_loss}

        return loss_dict_q1, loss_dict_q2

    def _critic_loss(
            self,
            key,
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
        if self.split_q and key == "main":  # Double headed Q function - one for the extrinsic reward and one for the intrinsic reward
            with torch.no_grad():
                # first next_actions from current policy,
                if markov_actor:
                    new_actions, new_mean, new_log_std, new_log_probs = actor[key](next_observs if markov_critic else observs)
                else:
                    # (T+1, B, dim) including reaction to last obs
                    new_actions, new_mean, new_log_std, new_log_probs = actor[key](
                        prev_actions=actions,
                        rewards=rewards,
                        observs=next_observs if markov_critic else observs,
                    )

                if markov_critic:  # (B, 1)
                    next_q1, next_q2 = critic_target[key](next_observs, new_actions)
                else:
                    next_q1, next_q2 = critic_target[key](
                        prev_actions=actions,
                        rewards=rewards,
                        observs=observs,
                        current_actions=new_actions,
                    )  # (T+1, B, A)
                next_q1_env, next_q1_teacher = next_q1
                next_q2_env, next_q2_teacher = next_q2
                min_next_q_env_target = torch.min(next_q1_env, next_q2_env)
                min_next_q_teacher_target = torch.min(next_q1_teacher, next_q2_teacher)

                if markov_critic:
                    min_next_q_teacher_target -= torch.norm(new_actions - teacher_next_log_probs, dim=1).unsqueeze(dim=1)  # (T+1, B, 1)
                else:
                    min_next_q_teacher_target -= torch.norm(new_actions - teacher_log_probs, dim=1).unsqueeze(dim=1)  # (T+1, B, 1)

                # q_target: (T, B, 1)
                q_teacher_target = (1.0 - dones) * gamma * min_next_q_teacher_target  # next q TODO: do we need the 1-done here?
                q_env_target = rewards + (1.0 - dones) * gamma * min_next_q_env_target  # next q

                if not markov_critic:
                    q_teacher_target = q_teacher_target[1:]  # (T, B, 1)
                    q_env_target = q_env_target[1:]  # (T, B, 1)

            if markov_critic:
                q1_pred, q2_pred = critic[key](observs, actions)
                q1_env_pred, q1_teacher_pred = q1_pred
                q2_env_pred, q2_teacher_pred = q2_pred
                qf1_teacher_loss = F.mse_loss(q1_teacher_pred, q_teacher_target)  # TD error
                qf2_teacher_loss = F.mse_loss(q2_teacher_pred, q_teacher_target)  # TD error
                qf1_env_loss = F.mse_loss(q1_env_pred, q_env_target)  # TD error
                qf2_env_loss = F.mse_loss(q2_env_pred, q_env_target)  # TD error

            else:
                # Q(h(t), a(t)) (T, B, 1)
                q1_pred, q2_pred = critic[key](
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=actions[1:],
                )  # (T, B, A)
                q1_env_pred, q1_teacher_pred = q1_pred
                q2_env_pred, q2_teacher_pred = q2_pred

                # masked Bellman error: masks (T,B,1) ignore the invalid error
                # this is not equal to masks * q1_pred, cuz the denominator in mean()
                # 	should depend on masks > 0.0, not a constant B*T
                assert 'masks' in kwargs
                masks = kwargs['masks']
                num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
                q1_teacher_pred, q2_teacher_pred = q1_teacher_pred * masks, q2_teacher_pred * masks
                q1_env_pred, q2_env_pred = q1_env_pred * masks, q2_env_pred * masks
                q_teacher_target = q_teacher_target * masks
                q_env_target = q_env_target * masks
                qf1_teacher_loss = ((q1_teacher_pred - q_teacher_target) ** 2).sum() / num_valid  # TD error
                qf2_teacher_loss = ((q2_teacher_pred - q_teacher_target) ** 2).sum() / num_valid  # TD error
                qf1_env_loss = ((q1_env_pred - q_env_target) ** 2).sum() / num_valid  # TD error
                qf2_env_loss = ((q2_env_pred - q_env_target) ** 2).sum() / num_valid  # TD error

            return (qf1_teacher_loss, qf1_env_loss), (qf2_teacher_loss, qf2_env_loss)
        else:  # Single headed (regular) Q function
            raise NotImplementedError

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
        main_policy_loss, main_additional_ouputs = self._actor_loss(
            "main",
            markov_actor=markov_actor,
            markov_critic=markov_critic,
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            states=states,
            teacher_log_probs=teacher_log_probs,
            **kwargs
        )
        if self.split_q:
            loss_dict = {"main_loss_q": main_policy_loss[0], "main_loss_t": main_policy_loss[1]}
        else:
            loss_dict = {"main_loss": main_policy_loss}
        if self.coefficient_tuning in ["EIPO"]:
            aux_policy_loss, aux_additional_ouputs = self._actor_loss(
                "aux",
                markov_actor=markov_actor,
                markov_critic=markov_critic,
                actor=actor,
                actor_target=actor_target,
                critic=critic,
                critic_target=critic_target,
                observs=observs,
                actions=actions,
                rewards=rewards,
                states=states,
                teacher_log_probs=teacher_log_probs,
                **kwargs
            )
            loss_dict["aux_loss"] = aux_policy_loss
        return loss_dict, main_additional_ouputs

    def _actor_loss(
            self,
            key,
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
        if self.split_q and key == "main":
            if markov_actor:
                new_actions, mean, log_std, log_probs = actor[key](observs)
            else:
                new_actions, mean, log_std, log_probs = actor[key](
                    prev_actions=actions, rewards=rewards, observs=observs
                )  # (T+1, B, A).

            if markov_critic:  # (B, A)
                next_q1, next_q2 = critic_target[key](observs, new_actions)
            else:
                next_q1, next_q2 = critic_target[key](
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_actions,
                )  # (T+1, B, 1)
            next_q1_env, next_q1_teacher = next_q1
            next_q2_env, next_q2_teacher = next_q2
            min_next_q_env_target = torch.min(next_q1_env, next_q2_env)
            min_next_q_teacher_target = torch.min(next_q1_teacher, next_q2_teacher)

            policy_loss_q = -min_next_q_env_target
            policy_loss_q -= self.coefficient * min_next_q_teacher_target
            policy_loss_q += log_probs
            policy_loss_t = self.coefficient * torch.norm(new_actions - teacher_log_probs, dim=1).unsqueeze(dim=1)

            if not markov_critic:
                policy_loss_q = policy_loss_q[:-1]  # (T,B,1) remove the last obs
                policy_loss_t = policy_loss_t[:-1]  # (T,B,1) remove the last obs

            if not markov_actor:
                assert 'masks' in kwargs
                masks = kwargs['masks']
                num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
                policy_loss_q = (policy_loss_q * masks).sum() / num_valid
                policy_loss_t = (policy_loss_t * masks).sum() / num_valid

            additional_outputs = {}
            if key == "main":
                # -> negative entropy (T+1, B, 1)
                additional_outputs['negative_entropy'] = (log_probs).sum(axis=-1, keepdims=True)
                additional_outputs['negative_cross_entropy'] = (teacher_log_probs).sum(axis=-1,
                                                                                                   keepdims=True)
            return (policy_loss_q, policy_loss_t), additional_outputs
        if self.split_q and key == "aux":
            if markov_actor:
                new_actions, mean, log_std, log_probs = actor[key](observs)
            else:
                new_actions, mean, log_std, log_probs = actor[key](
                    prev_actions=actions, rewards=rewards, observs=observs
                )  # (T+1, B, A).

            if markov_critic:  # (B, A)
                next_q1, next_q2 = critic_target["main"](observs, new_actions)
            else:
                next_q1, next_q2 = critic_target["main"](
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_actions,
                )  # (T+1, B, A)
            next_q1_env, next_q1_teacher = next_q1
            next_q2_env, next_q2_teacher = next_q2
            min_next_q_env_target = torch.min(next_q1_env, next_q2_env)

            policy_loss = -min_next_q_env_target

            if not markov_critic:
                policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs

            if not markov_actor:
                assert 'masks' in kwargs
                masks = kwargs['masks']
                num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
                policy_loss = (policy_loss * masks).sum() / num_valid

            return policy_loss, None
        else:
            raise NotImplementedError

    def update_others(self, additional_outputs, markov_critic, markov_actor, critic, actor, observs, actions, rewards, reward_std):
        assert 'negative_entropy' in additional_outputs
        assert 'negative_cross_entropy' in additional_outputs
        current_entropy = -additional_outputs['negative_entropy'].mean().item()
        current_cross_entropy = -additional_outputs['negative_cross_entropy'].mean().item()

        # # Update statistics
        # self.cross_entropy_sum += (-additional_outputs['negative_cross_entropy']).sum().item()
        # self.cross_entropy_sumsq += ((-additional_outputs['negative_cross_entropy'])**2).sum().item()
        # self.cross_entropy_count += len(-additional_outputs['negative_cross_entropy'])
        # self.cross_entropy_mean = self.cross_entropy_sum / self.cross_entropy_count
        # self.cross_entropy_std = sqrt((self.cross_entropy_sumsq/self.cross_entropy_count) - (self.cross_entropy_mean*self.cross_entropy_mean))

        objective_difference = 0.0
        normalized_obj_difference = 0.0
        if self.coefficient_tuning == "Target":
            coefficient_loss = self.log_coefficient.exp() * (current_cross_entropy - self.target_coefficient)
            self.coefficient_optim.zero_grad()
            coefficient_loss.backward()
            self.coefficient_optim.step()
            self.coefficient = self.log_coefficient.exp().item()
        if self.coefficient_tuning == "EIPO":
            objective_difference = self.estimate_objective_difference()  # J(pi_{E+I}) - J(pi_{E})
            normalized_obj_difference = objective_difference / reward_std
            self.log_coefficient = torch.clip(self.log_coefficient + self.coefficient_lr * normalized_obj_difference, np.log(self.min_coefficent),
                                              np.log(self.max_coefficent))
            self.coefficient = self.log_coefficient.exp().item()

        output_dict = {"cross_entropy": current_cross_entropy,
                       "policy_entropy": current_entropy,
                       "coefficient": self.coefficient}
        if self.coefficient_tuning == "EIPO":
            output_dict["objective_difference"] = objective_difference
            output_dict["normalized_objective_difference"] = normalized_obj_difference
            # output_dict["objective_difference_approx"] = obj_aproximation
            output_dict["obj_est_main"] = self.obj_est_main
            output_dict["obj_est_aux"] = self.obj_est_aux
        return output_dict

    def get_acting_policy_key(self):
        return self.current_policy

    def estimate_objective_difference(self):
        return self.obj_est_main - self.obj_est_aux

