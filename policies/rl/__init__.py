from .td3 import TD3
from .sac import SAC
from .sacd import SACD
from .eaacd import EAACD

RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
    SACD.name: SACD,
    EAACD.name: EAACD
}
