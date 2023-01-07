from .td3 import TD3
from .sac import SAC
from .sacd import SACD
from .eaacd import EAACD
from .dagger import DAgger
from .ADVISORd import ADVISORd
from .ELFd import ELFd
from .daggerc import DAggerc
from .eaac import EAAC

RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
    SACD.name: SACD,
    EAACD.name: EAACD,
    DAgger.name: DAgger,
    ADVISORd.name: ADVISORd,
    ELFd.name: ELFd,
    DAggerc.name: DAggerc,
    EAAC.name: EAAC
}
