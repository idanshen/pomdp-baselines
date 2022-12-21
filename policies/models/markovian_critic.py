import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *


class Critic_Markovian(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        algo,
        dqn_layers,
        critic_key,
        image_encoder=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo

        ### Build Model
        ## 1. embed observations

        self.image_encoder = image_encoder
        if self.image_encoder is None:
            observ_embedding_size = obs_dim
        else:  # for pixel observation, use external encoder
            observ_embedding_size = self.image_encoder.embed_size  # reset it

        ## 2. build q networks

        self.q_funcs = self.algo.build_critic(
            obs_dim=observ_embedding_size,
            hidden_sizes=dqn_layers,
            action_dim=action_dim,
        )
        self.qf1 = self.q_funcs[critic_key + "_qf1"]
        self.qf2 = self.q_funcs[critic_key + "_qf2"]

    def _get_obs_embedding(self, observs):
        if self.image_encoder is None:  # vector obs
            return observs
        else:  # pixel obs
            return self.image_encoder(observs)

    def forward(self, observs, *inputs):
        embedded_observs = self._get_obs_embedding(observs)

        q1 = self.qf1(embedded_observs)
        q2 = self.qf2(embedded_observs)

        return q1, q2
