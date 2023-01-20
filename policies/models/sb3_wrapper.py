import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu
import numpy as np

class SB3_Wrapper(nn.Module):
    def __init__(
        self,
        sb3_model,
        **kwargs
    ):
        super().__init__()

        self.model = sb3_model

    def forward(self, observs):
        actions, _ = self.model.predict(
            observs.cpu().numpy(),  # type: ignore[arg-type]
            state=None,
            episode_start=None,
            deterministic=False,
        )

        return actions

    def act(self, obs, deterministic=False, return_log_prob=False,):
        actions, _ = self.model.predict(
            obs,  # type: ignore[arg-type]
            state=None,
            episode_start=None,
            deterministic=deterministic,
        )
        return actions, None, None, None