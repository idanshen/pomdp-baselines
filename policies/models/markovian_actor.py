import torch
import torch.nn as nn
from torch.nn import functional as F

from policies.rl import A2D
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu


class Actor_Markovian(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        algo,
        policy_layers,
        policy_key,
        image_encoder=None,
        **kwargs
    ):
        super().__init__()

        self.action_dim = action_dim
        self.algo = algo
        self.obs_dim = obs_dim
        if type(algo) is A2D and policy_key == "aux":
            self.obs_dim = kwargs['state_dim']

        ### Build Model
        ## 1. embed observations

        self.image_encoder = image_encoder
        if self.image_encoder is None:
            observ_embedding_size = self.obs_dim
        else:  # for pixel observation, use external encoder
            observ_embedding_size = self.image_encoder.embed_size  # reset it

        ## 4. build policy
        self.policy = self.algo.build_actor(
            input_size=observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
        )[policy_key+"_actor"]

    def _get_obs_embedding(self, observs):
        if self.image_encoder is None:  # vector obs
            return observs
        else:  # pixel obs
            return self.image_encoder(observs)

    def forward(self, observs):
        embedded_observs = self._get_obs_embedding(observs)
        actions = self.algo.forward_actor(actor=self.policy, observ=embedded_observs)

        return actions

    def act(self, obs, deterministic=False, return_log_prob=False,):
        embedded_observs = self._get_obs_embedding(obs)

        return self.algo.select_action(
                actor=self.policy,
                observ=embedded_observs,
                deterministic=deterministic,
                return_log_prob=return_log_prob,
            )