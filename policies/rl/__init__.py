from .td3 import TD3
from .sac import SAC
from .sacd import SACD
from .eaacd import EAACD
from .dagger import DAgger
from .ADVISORd import ADVISORd
from .ELFd import ELFd

RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
    SACD.name: SACD,
    EAACD.name: EAACD,
    DAgger.name: DAgger,
    ADVISORd.name: ADVISORd,
    ELFd.name: ELFd,
}
