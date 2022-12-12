import torch
import torch.nn as nn
from torch.nn import functional as F
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

        ## 4. build policy
        self.policies = self.algo.build_actor(
            input_size=observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
        )

    def _get_obs_embedding(self, observs):
        if self.image_encoder is None:  # vector obs
            return observs
        else:  # pixel obs
            return self.image_encoder(observs)

    def forward(self, observs):
        embedded_observs = self._get_obs_embedding(observs)

        actions = {}
        for key, policy in self.policies.items():
            actions[key] = self.algo.forward_actor(actor=policy, observ=embedded_observs)

        return actions

    def act(self, obs, deterministic=False, return_log_prob=False,):
        acting_policy_key = self.algo.get_acting_policy_key()
        embedded_observs = self._get_obs_embedding(obs)

        return self.algo.select_action(
                actor=self.policies[acting_policy_key],
                observ=embedded_observs,
                deterministic=deterministic,
                return_log_prob=return_log_prob,
            )