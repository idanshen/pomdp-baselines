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
        pass
        # self.teacher = self.load_teacher(teacher_dir, state_dim=state_dim, act_dim=action_dim)

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
            new_probs, log_probs = self.forward_actor(actor, observs)
        else:
            new_probs, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)
        # with torch.no_grad():
        #     # only markov teacher supported
        #     _, teacher_probs, teacher_log_probs = self.teacher(states, return_log_prob=True)

        # Imitation
        if markov_actor:
            policy_loss = torch.norm(new_probs - torch.exp(teacher_actions), dim=1).unsqueeze(dim=1)
        else:
            policy_loss = torch.norm(new_probs[:-1] - torch.exp(teacher_actions[:-1]), dim=2).unsqueeze(dim=2)

        # -> negative entropy (T+1, B, 1)
        with torch.no_grad():
            log_probs_est = (new_probs * log_probs).sum(axis=-1, keepdims=True)
            # if markov_actor:
            #     log_probs_est = torch.all(new_probs.round() == torch.exp(teacher_actions).round(), dim=1).type(torch.float32).unsqueeze(dim=-1)
            # else:
            #     log_probs_est = torch.all(new_probs.round()[:-1] == torch.exp(teacher_actions[:-1]).round(), dim=2).type(torch.float32).unsqueeze(dim=-1)

        # policy_loss = self.external_loss(actor, policy_loss)
        return policy_loss, log_probs_est

    def update_others(self, current_log_probs):
        return {"policy_entropy": -current_log_probs}
        # return {"imitation accuracy": current_log_probs}

    def external_loss(self, actor, policy_loss):
        tmp = np.load("/home/idanshen/tmp/seq_data.npy")
        observs = ptu.from_numpy(tmp[:, :, :30])
        labels = ptu.from_numpy(tmp[:, :, 30:34])
        labels = torch.exp(labels)
        mask = ptu.from_numpy(tmp[:, :, 34:35])
        rewards = ptu.from_numpy(tmp[:, :, 35:36])
        prev_actions_class = ptu.from_numpy(tmp[:, :, 36:])
        prev_actions = torch.zeros_like(labels)
        prev_actions[prev_actions_class.squeeze() == 0.0] = ptu.from_numpy(np.array([1., 0., 0., 0.]))
        prev_actions[prev_actions_class.squeeze() == 1.0] = ptu.from_numpy(np.array([0., 1., 0., 0.]))
        prev_actions[prev_actions_class.squeeze() == 2.0] = ptu.from_numpy(np.array([0., 0., 1., 0.]))
        prev_actions[prev_actions_class.squeeze() == 3.0] = ptu.from_numpy(np.array([0., 0., 0., 1.]))

        observs = torch.cat((observs[[0]], observs[1:], observs[[0]]), dim=0)  # (T+1, B, dim)
        prev_actions = torch.cat(
            (ptu.zeros((1, 128, 4)).float(), prev_actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, 128, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)

        prob, log_probs = actor(prev_actions, rewards, observs)
        policy_loss = torch.norm(prob[:-1] - labels, dim=2).unsqueeze(dim=2)
        masked_policy_loss = policy_loss * mask
        policy_loss = masked_policy_loss.mean()
        return policy_loss
