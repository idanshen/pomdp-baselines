import torch
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
import torch.nn.functional as F
from utils import helpers as utl


class A2D(RLAlgorithmBase):
    name = "a2d"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        entropy_alpha=0.1,
        action_dim=None,
        state_dim=None,
        teacher_initial_weights=None,
        **kwargs
    ):
        super().__init__()
        self.alpha_entropy = entropy_alpha
        if teacher_initial_weights is not None:
            self.teacher_initial_policy, _ = utl.load_teacher(teacher_initial_weights, state_dim=state_dim, act_dim=action_dim)

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
            min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, 1)

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
        next_observs=None,
        states=None,
        dones=None,
        **kwargs,
    ):
        if markov_actor:
            new_main_actions, _, _, _ = actor["main"](observs)  # main is student
            new_aux_actions, _, _, aux_log_probs = actor["aux"](states)  # aux is teacher
        else:
            raise NotImplementedError

        if markov_critic:
            q1, q2 = critic["main"](observs, new_aux_actions)
        else:
            raise NotImplementedError
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * aux_log_probs
        imitation_loss = torch.norm(new_main_actions - new_aux_actions.detach(), dim=-1).unsqueeze(dim=-1)
        if not markov_critic:
            raise NotImplementedError
        if not markov_actor:
            raise NotImplementedError

        additional_outputs = {}
        # -> negative entropy (T+1, B, 1)
        additional_outputs['negative_entropy'] = aux_log_probs

        return {"aux_loss": policy_loss, "main_loss": imitation_loss}, additional_outputs

    def update_others(self, additional_outputs, **kwargs):
        assert 'negative_entropy' in additional_outputs
        current_log_probs = additional_outputs['negative_entropy'].mean().item()
        return {"teacher_policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}

    def get_acting_policy_key(self):
        return self.current_policy

    @property
    def model_keys(self):
        return {"actor": ["main", "aux"], "critic": ["main"]}