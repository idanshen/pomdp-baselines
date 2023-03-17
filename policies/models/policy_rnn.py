""" Recommended Architecture
Separate RNN architecture is inspired by a popular RL repo
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/POMDP/common/value_networks.py#L110
which has another branch to encode current state (and action)

Hidden state update functions get_hidden_state() is inspired by varibad encoder 
https://github.com/lmzintgraf/varibad/blob/master/models/encoder.py
"""

import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

from policies.models.recurrent_value import Value_RNN
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_critic import Critic_RNN
from policies.models.recurrent_actor import Actor_RNN
from utils import logger


class ModelFreeOffPolicy_Separate_RNN(nn.Module):
    """Recommended Architecture
    Recurrent Actor and Recurrent Critic with separate RNNs
    """

    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo_name,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        policy_layers,
        rnn_num_layers=1,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        alpha=0.5,
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
        self.alpha = alpha
        self.algo_name = algo_name

        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name],
                                             teacher_dir=teacher_dir,
                                             action_dim=action_dim,
                                             state_dim=self.state_dim,
                                             obs_dim=self.obs_dim)
        self.critic = torch.nn.ModuleDict({key: Critic_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            rnn_num_layers,
            key,
            image_encoder=image_encoder_fn(),  # separate weight
        ) for key in self.algo.model_keys["critic"]})
        # Critics

        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        # target networks
        self.critic_target = deepcopy(self.critic)

        # Actor
        self.actor = torch.nn.ModuleDict({key: Actor_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            policy_layers,
            rnn_num_layers,
            key,
            image_encoder=image_encoder_fn(),  # separate weight
        ) for key in self.algo.model_keys["actor"]})
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        # target networks
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def get_initial_info(self, key, batch_size):
        return self.actor["main"].get_initial_info(batch_size)

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, dim)

        policy_key = self.algo.get_acting_policy_key()
        if policy_key == "main":
            curr_actor = self.actor["main"]
            current_action_tuple, current_internal_state = curr_actor.act(
                prev_internal_state=prev_internal_state,
                prev_action=prev_action,
                reward=reward,
                obs=obs,
                deterministic=deterministic,
                return_log_prob=return_log_prob,
            )
        else:
            if self.algo_name == "eaacd":
                current_action_tuple, current_internal_state = self.algo.aux_act(
                    critic=self.critic,
                    markov_critic=False,
                    prev_internal_state=prev_internal_state,
                    prev_action=prev_action,
                    reward=reward,
                    obs=obs,
                    deterministic=deterministic,
                    return_log_prob=return_log_prob,
                )
            else:
                curr_actor = self.actor["aux"]
                current_action_tuple, current_internal_state = curr_actor.act(
                    prev_internal_state=prev_internal_state,
                    prev_action=prev_action,
                    reward=reward,
                    obs=obs,
                    deterministic=deterministic,
                    return_log_prob=return_log_prob,
                )

        return current_action_tuple, current_internal_state

    def compute_loss(self, actions, rewards, observs, dones, masks, states=None, teacher_log_probs=None, reward_mean=None, reward_std=None):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t] and state s[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
        outputs = {}

        ### 1. Critic loss
        qf1_losses, qf2_losses = self.algo.critic_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            states=states,
            masks=masks,
            teacher_log_probs=teacher_log_probs
        )

        self.critic_optimizer.zero_grad()
        qf1_loss = 0
        for key, loss in qf1_losses.items():
            qf1_loss += loss
            outputs["qf1_"+key] = loss.item()
        qf2_loss = 0
        for key, loss in qf2_losses.items():
            qf2_loss += loss
            outputs["qf2_" + key] = loss.item()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

        ### 2. Actor loss
        policy_losses, additional_outputs = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            states=states,
            teacher_log_probs=teacher_log_probs,
            dones=dones,
            masks=masks
        )

        self.actor_optimizer.zero_grad()
        policy_loss = 0
        for key, loss in policy_losses.items():
            policy_loss += loss.mean()
            outputs["policy_" + key] = loss.mean().item()
        policy_loss.backward()
        self.actor_optimizer.step()

        ### 3. soft update
        self.soft_target_update()

        ### 4. update others like alpha
        if additional_outputs is not None:
            # extract valid log_probs
            for k, v in additional_outputs.items():
                with torch.no_grad():
                    value = (v[:-1] * masks).sum() / num_valid
                    additional_outputs[k] = value

            other_info = self.algo.update_others(additional_outputs, markov_critic=self.Markov_Critic, markov_actor=self.Markov_Actor, critic=self.critic,
                                                 actor=self.actor, observs=observs, actions=actions, rewards=rewards, reward_std=reward_std)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic["main"]),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic["main"].rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor["main"]),
            "pi_rnn_grad_norm": utl.get_grad_norm(self.actor["main"].rnn),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        if "states" in batch:
            states, next_states = batch["states"], batch["states2"]
            states = torch.cat((states[[0]], next_states), dim=0)
        else:
            states = None
        teacher_log_probs = batch["teacher_log_prob"]
        teacher_next_log_probs = batch["teacher_log_prob2"]
        teacher_log_probs = torch.cat((teacher_log_probs[[0]], teacher_next_log_probs), dim=0)  # (T+1, B, dim)
        reward_mean, reward_std = batch["rew_mean"], batch["rew_std"]
        return self.compute_loss(actions, rewards, observs, dones, masks, states, teacher_log_probs, reward_mean, reward_std)
