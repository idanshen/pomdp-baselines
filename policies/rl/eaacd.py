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
import torch.nn.functional as F


class EAACD(RLAlgorithmBase):
    name = "eaacd"
    continuous_action = False
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
    ):
        assert coefficient_tuning in ["Fixed", "Target", "EIPO"]
        self.coefficient_tuning = coefficient_tuning
        self.current_policy = "main"
        self.obj_est_main = 0.0
        self.obj_est_aux = 0.0

        if self.coefficient_tuning == "Target":
            assert target_coefficient is not None
            self.target_coefficient = float(target_coefficient)
            self.log_coefficient = torch.zeros(1, requires_grad=True, device=ptu.device)
            self.coefficient_optim = Adam([self.log_coefficient], lr=coefficient_lr)
            self.coefficient = self.log_coefficient.exp().detach().item()
        elif self.coefficient_tuning == "Fixed":
            self.coefficient = initial_coefficient
        else:
            self.log_coefficient = torch.tensor(np.log(initial_coefficient), requires_grad=True, device=ptu.device)
            self.coefficient = self.log_coefficient.exp().detach().item()
            self.coefficient_lr = coefficient_lr / 500.0
        self.teacher = self.load_teacher(teacher_dir, state_dim=state_dim, act_dim=action_dim)

    def build_actor(self, input_size, action_dim, hidden_sizes, **kwargs):
        if type(input_size) == tuple:
            assert len(input_size) == 1
            input_size = input_size[0]

        main_actor = CategoricalPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
        actors = nn.ModuleDict({"main_actor": main_actor})
        if self.coefficient_tuning == "EIPO":
            aux_actor = CategoricalPolicy(
                obs_dim=input_size,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                **kwargs,
            )
            actors["aux_actor"] = aux_actor
        return actors

    def build_critic(self, hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        if type(input_size) == tuple:
            assert len(input_size) == 1
            input_size = input_size[0]
        main_qf1 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        main_qf2 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        qfs = nn.ModuleDict({"main_qf1": main_qf1, "main_qf2": main_qf2})
        if self.coefficient_tuning == "EIPO":
            aux_qf1 = FlattenMlp(
                input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
            )
            aux_qf2 = FlattenMlp(
                input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
            )
            qfs["aux_qf1"] = aux_qf1
            qfs["aux_qf2"] = aux_qf2
        return qfs

    @staticmethod
    def load_teacher(teacher_dir: str, state_dim, act_dim):
        assert teacher_dir is not None
        files = glob.glob(teacher_dir + "*.yml")
        assert len(files) == 1
        config_file = files[0]
        yaml = YAML()
        v = yaml.load(open(config_file))

        agent_class, rnn_encoder_type = utl.parse_seq_model(
            v['policy']['seq_model'],
            v['policy']['seq_model'] if 'seperate' in v['policy'] else True)
        teacher = agent_class(
            encoder=rnn_encoder_type,
            obs_dim=state_dim,
            action_dim=act_dim,
            image_encoder_fn=lambda: None,
            **v['policy'],
        ).to(ptu.device)
        models = glob.glob(teacher_dir + "save/*")
        model_path = sorted(models)[-1]
        teacher.load_state_dict(torch.load(model_path, map_location=ptu.device))
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher.policy

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
        next_states=None
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
                          next_states=next_states)
        if self.coefficient_tuning == "EIPO":
            aux_qf1_loss, aux_qf2_loss = self._critic_loss("aux",
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
                              next_states=next_states)
            return {"main_loss": main_qf1_loss, "aux_loss": aux_qf1_loss}, {"main_loss": main_qf2_loss, "aux_loss": aux_qf2_loss}
        else:
            return {"main_loss": main_qf1_loss}, {"main_loss": main_qf2_loss}

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
        next_states=None
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from current policy,
            if markov_actor:
                new_probs, new_log_probs = actor(next_observs if markov_critic else observs)[key+"_actor"]
            else:
                # (T+1, B, dim) including reaction to last obs
                new_probs, new_log_probs = actor(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                )[key+"_actor"]

            if key == "main":
                # only markov teacher supported
                # (T+1, B, dim) including reaction to last obs
                if markov_actor:
                    _, teacher_probs, teacher_log_probs, _ = self.teacher.act(next_states, return_log_prob=True)
                else:
                    _, teacher_probs, teacher_log_probs, _ = self.teacher.act(states, return_log_prob=True)

            if markov_critic:  # (B, A)
                q_dict = critic_target(next_observs)
                next_q1, next_q2 = q_dict[key+"_qf1"], q_dict[key+"_qf2"]
            else:
                q_dict = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_probs,
                )  # (T+1, B, A)
                next_q1, next_q2 = q_dict[key + "_qf1"], q_dict[key + "_qf2"]

            min_next_q_target = torch.min(next_q1, next_q2)

            if key == "main":
                min_next_q_target += self.coefficient * (teacher_log_probs)  # (T+1, B, A)

            # E_{a'\sim \pi}[Q(h',a')], (T+1, B, 1)
            min_next_q_target = (new_probs * min_next_q_target).sum(
                dim=-1, keepdims=True
            )

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        if markov_critic:
            q_pred_dict = critic(observs)
            q1_pred, q2_pred = q_pred_dict[key+"_qf1"], q_pred_dict[key+"_qf2"]
            action = actions.long()  # (B, 1)
            q1_pred = q1_pred.gather(dim=-1, index=action)
            q2_pred = q2_pred.gather(dim=-1, index=action)

        else:
            # Q(h(t), a(t)) (T, B, 1)
            q_pred_dict = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
            )  # (T, B, A)
            q1_pred, q2_pred = q_pred_dict[key + "_qf1"], q_pred_dict[key + "_qf2"]

            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdims=True
            )  # (T, B, 1)
            q1_pred = q1_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
            q2_pred = q2_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        return qf1_loss, qf2_loss

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
        )
        if self.coefficient_tuning == "EIPO":
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
            )
            return {"main_loss": main_policy_loss, "aux_loss": aux_policy_loss}, main_additional_ouputs
        else:
            return {"main_loss": main_policy_loss}, main_additional_ouputs

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
    ):
        if markov_actor:
            new_probs, log_probs = actor(observs)[key+"_actor"]
        else:
            new_probs, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )[key+"_actor"]  # (T+1, B, A)

        if markov_critic:
            q_dict = critic(observs)
            q1, q2 = q_dict[key+"_qf1"], q_dict[key+"_qf2"]
        else:
            q_dict = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_probs,
            )  # (T+1, B, A)
            q1, q2 = q_dict[key + "_qf1"], q_dict[key + "_qf2"]
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,A)

        policy_loss = -min_q_new_actions
        policy_loss += log_probs
        if key == "main":
            policy_loss -= self.coefficient * teacher_log_probs

        # E_{a\sim \pi}[Q(h,a)]
        policy_loss = (new_probs * policy_loss).sum(axis=-1, keepdims=True)  # (T+1,B,1)
        if not markov_critic:
            policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs

        additional_outputs = {}
        if key == "main":
            # -> negative entropy (T+1, B, 1)
            additional_outputs['negative_entropy'] = (new_probs * log_probs).sum(axis=-1, keepdims=True)
            additional_outputs['negative_cross_entropy'] = (new_probs * teacher_log_probs).sum(axis=-1, keepdims=True)

        return policy_loss, additional_outputs

    def update_others(self, additional_outputs):
        assert 'negative_entropy' in additional_outputs
        assert 'negative_cross_entropy' in additional_outputs
        current_entropy = -additional_outputs['negative_entropy'].mean().item()
        current_cross_entropy = -additional_outputs['negative_cross_entropy'].mean().item()

        if self.coefficient_tuning == "Target":
            coefficient_loss = self.log_coefficient.exp() * (current_cross_entropy - self.target_coefficient)
            self.coefficient_optim.zero_grad()
            coefficient_loss.backward()
            self.coefficient_optim.step()
            self.coefficient = self.log_coefficient.exp().item()
        objective_difference = 0.0
        if self.coefficient_tuning == "EIPO":
            objective_difference = self.estimate_objective_difference()  # J(pi_{E+I}) - J(pi_{E})
            self.log_coefficient = torch.clip(self.log_coefficient + self.coefficient_lr * objective_difference, -5, 5)
            self.coefficient = self.log_coefficient.exp().item()
        return {"cross_entropy": current_cross_entropy,
                "policy_entropy": current_entropy,
                "coefficient": self.coefficient,
                "objective_difference": objective_difference,
                "obj_est_main": self.obj_est_main,
                "obj_est_aux": self.obj_est_aux}

    def get_acting_policy_key(self):
        return self.current_policy+"_actor"

    def estimate_objective_difference(self):
        return self.obj_est_main - self.obj_est_aux
